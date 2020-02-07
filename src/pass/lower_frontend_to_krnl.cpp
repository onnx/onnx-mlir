//====- lower_frontend_to_krnl.cpp - Frontend dialects to Krnl lowering ---===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to a combination of
// Krnl IR and standard operations.
//
//===----------------------------------------------------------------------===//
#include <map>

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"

#include "src/dialect/krnl/krnl_helper.hpp"
#include "src/dialect/krnl/krnl_ops.hpp"
#include "src/dialect/onnx/onnx_ops.hpp"
#include "src/pass/passes.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// FrontendToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Check is all dimensions are known at compile time.
static bool hasAllConstantDimensions(MemRefType type) {
  auto memRefShape = type.getShape();
  for (int i = 0; i < memRefShape.size(); ++i)
    if (memRefShape[i] < 0)
      return false;
  return true;
}

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter,
                                   bool insertDealloc,
                                   ArrayRef<Value> operands = {}) {
  // Put together alloc operands for any dynamic dimensions of the memref.
  AllocOp alloc;
  if (!operands.empty()) {
    auto memRefShape = type.getShape();
    auto rank = memRefShape.size();

    std::map<int, Value> fromOperands;
    for (int reversedIdx = 0; reversedIdx < rank; ++reversedIdx) {
      int memRefDimIdx = rank - 1 - reversedIdx;
      if (memRefShape[memRefDimIdx] < 0) { // unknown dimension
        Value maxDim = nullptr;
        for (int i = 0; i < operands.size(); i++) {
          auto operandShape =
              operands[i].getType().cast<MemRefType>().getShape();
          int operandDimIdx = operandShape.size() - 1 - reversedIdx;

          if (operandDimIdx < 0)
            continue;

          // In case of operations with broadcasting, the dimension of the
          // alloc result is the maximum size along each dimension of the
          // operands.
          auto operandDim =
              rewriter.create<DimOp>(loc, operands[i], operandDimIdx);
          if (maxDim) {
            auto maxCondition = rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt,
                                                        operandDim, maxDim);
            maxDim = rewriter.create<SelectOp>(loc, maxCondition, operandDim,
                                               maxDim);
          } else {
            maxDim = operandDim;
          }
        }
        fromOperands.insert(std::make_pair(memRefDimIdx, maxDim));
      }
    }

    SmallVector<Value, 4> allocOperands;
    for (int i = 0; i < rank; ++i)
      if (memRefShape[i] < 0)
        allocOperands.push_back(fromOperands[i]);
    alloc = rewriter.create<AllocOp>(loc, type, allocOperands);
  } else {
    alloc = rewriter.create<AllocOp>(loc, type);
  }

  // Make sure to allocate at the beginning of the block if
  // all dimensions are known.
  auto *parentBlock = alloc.getOperation()->getBlock();
  if (hasAllConstantDimensions(type))
    alloc.getOperation()->moveBefore(&parentBlock->front());

  if (insertDealloc) {
    auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }

  return alloc;
}

// Determine if current function returns the result value of the
// current op being lowered. If it does then dealloc should not be
// inserted.
static bool checkInsertDealloc(Operation *currentOp) {
  auto parentBlock = currentOp->getBlock();

  bool insertDealloc = true;
  parentBlock->walk([&insertDealloc, currentOp](ReturnOp op) {
    assert(currentOp->getNumResults() < 2 &&
           "No more than one result supported (for now).");
    // If there is at least one result to investigate.
    if (currentOp->getNumResults() > 0) {
      auto result = currentOp->getResult(0);
      for (const auto &operand : op.getOperands())
        if (operand == result)
          insertDealloc = false;
    }
  });

  return insertDealloc;
}

// Add bounds associated with the op operand to the KRNL iteration pack.
// Dynamic dimenions are supported.
static void addDimensionToPack(ConversionPatternRewriter &rewriter,
    Location loc, KrnlIterateOperandPack &pack, Value operand, int index) {
  auto shape = operand.getType().cast<MemRefType>().getShape();
  if (shape[index] < 0) {
    pack.pushConstantBound(0);
    pack.pushOperandBound(
        rewriter.create<DimOp>(loc, operand, index).getResult());
  } else {
    pack.pushConstantBound(0);
    pack.pushConstantBound(shape[index]);
  }
}

// Function that defines the KRNL dialect loops and their respective
// optimized version.
static KrnlOptimizeLoopsOp emitOptimizedLoops(
    ConversionPatternRewriter &rewriter, Location loc,
    std::vector<Value> &loops, std::vector<Value> &optimizedLoops,
    int64_t numLoops) {
  // Define loops.
  auto loopsOp = rewriter.create<KrnlDefineLoopsOp>(loc, numLoops);
  loops.reserve(numLoops);
  for (auto result : loopsOp.getResults())
    loops.push_back(result);

  // Define optimized version of the loops.
  auto optimizedLoopsOp = rewriter.create<KrnlOptimizeLoopsOp>(loc, numLoops);
  optimizedLoops.reserve(numLoops);
  for (auto result : optimizedLoopsOp.getResults())
    optimizedLoops.push_back(result);

  return optimizedLoopsOp;
}

// Function that emits the loops and their optimized version.
// The function returns a reference to the inner optimization block.
static Block* defineLoops(ConversionPatternRewriter &rewriter,
    Location loc, std::vector<Value> &loops,
    std::vector<Value> &optimizedLoops, int64_t numLoops) {
  KrnlOptimizeLoopsOp optimizedLoopsOp = emitOptimizedLoops(
      rewriter, loc, loops, optimizedLoops, numLoops);
  return &optimizedLoopsOp.region().front();
}

// Function which emits a basic set of loops and optimized loops
// for a given operation argument. A reference to the loop optimization
// block is returned in the last argument of the function.
static void emitKrnlLoopsAndIterationForOperand(
    ConversionPatternRewriter &rewriter, Location loc,
    Value operand, std::vector<Value> &originalLoops,
    KrnlOptimizeLoopsOp &optimizedLoopsOp, KrnlIterateOp &iterateOp) {
  // Operand shape.
  auto shape = operand.getType().cast<MemRefType>().getShape();

  // Number of loops.
  int64_t rank = shape.size();

  // Define loops and optimized loops.
  std::vector<Value> optimizedLoops;
  optimizedLoopsOp = emitOptimizedLoops(rewriter, loc, originalLoops,
      optimizedLoops, rank);

  KrnlIterateOperandPack pack(rewriter, originalLoops, optimizedLoops);
  // Iterate over the loop nest.
  for (int i = 0; i < rank; ++i)
    addDimensionToPack(rewriter, loc, pack, operand, i);

  iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
}

unsigned getMemRefEltSizeInBytes(MemRefType memRefType) {
  auto elementType = memRefType.getElementType();

  unsigned sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = elementType.cast<VectorType>();
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

// Get run-time dimension information for unknown dimensions used for
// broadcasting.
std::map<int, std::map<int, Value>>
getBroadcastedDimInfo(Location loc, ConversionPatternRewriter &rewriter,
                      MemRefType memRefType, ArrayRef<Value> operands) {
  auto memRefShape = memRefType.getShape();
  int64_t rank = memRefShape.size();
  // For unknown dimensions, we need to get dimension values at runtime in
  // order to do broadcasting.
  std::map<int, std::map<int, Value>> DimInfo;
  // For each result dimension, compute the number of sharing operands.
  // Sharing operands are operands sharing the same index (counting from the
  // rightmost to the leftmost) for a given dimension.
  std::map<int, int> sharedDimCount;
  for (int reversedIdx = 0; reversedIdx < rank; ++reversedIdx) {
    int dimIdx = rank - 1 - reversedIdx;
    sharedDimCount[dimIdx] = 0;
    for (int i = 0; i < operands.size(); ++i) {
      auto shape = operands[i].getType().cast<MemRefType>().getShape();
      if (reversedIdx <= shape.size() - 1)
        sharedDimCount[dimIdx]++;
    }
  }
  // An unknown dimension can have a value of 1 or N (N > 1).
  // If its value is 1, it is broadcasted dimension.
  // Otherwise, non-broadcasted dimension.
  // We only care about unknown dimensions whose number of sharing operands is
  // more than one, since they are potentially broadcasted dimensions.
  for (int i = 0; i < operands.size(); ++i) {
    std::map<int, Value> broadcastedDims;
    auto shape = operands[i].getType().cast<MemRefType>().getShape();
    int size = shape.size();
    for (int j = 0; j < shape.size(); ++j) {
      if (shape[j] < 0 and sharedDimCount[rank - size + j] > 1) {
        auto dim = rewriter.create<DimOp>(loc, operands[i], j).getResult();
        auto one = rewriter.create<ConstantIndexOp>(loc, 1);
        auto isBroadcasted =
          rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, dim, one);
        broadcastedDims.insert(std::make_pair(j, isBroadcasted));
      }
    }
    DimInfo.insert(std::make_pair(i, broadcastedDims));
  }
  return DimInfo;
}

// Extract induction variables that are used for broadcasting values of a
// given operand.
std::vector<Value>
getLoopIVsForBroadcasting(Location loc, ConversionPatternRewriter &rewriter,
                          ArrayRef<Value> loopIVs, Value operand,
                          std::map<int, Value> broadcastedDims) {
  // `operand` must has a ranked type. This should have been checked by the
  // shape inference pass.
  auto operandShape = operand.getType().cast<MemRefType>().getShape();
  auto rank = operandShape.size();
  auto loopCount = loopIVs.size();

  std::vector<Value> newLoopIVs;
  for (unsigned reversedIdx = 0; reversedIdx < rank; ++reversedIdx) {
    auto dimIdx = rank - 1 - reversedIdx;
    auto loopIdx = loopCount - 1 - reversedIdx;
    if (operandShape[dimIdx] == 1) {
      // Broadcasted dimension
      auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
      newLoopIVs.insert(newLoopIVs.begin(), zero);
    } else if ((operandShape[dimIdx] == -1) &&
               (broadcastedDims.find(dimIdx) != broadcastedDims.end())) {
      // Unknown dimension, it can have a value of 1 or N (N > 1).
      // If its value is 1, it is broadcasted dimension.
      // Otherwise, non-broadcasted dimension.
      auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
      auto idx = rewriter.create<SelectOp>(loc, broadcastedDims[dimIdx], zero,
                                           loopIVs[loopIdx]);
      newLoopIVs.insert(newLoopIVs.begin(), idx);
    } else {
      // Non-broadcasted dimension
      newLoopIVs.insert(newLoopIVs.begin(), loopIVs[loopIdx]);
    }
  }
  return newLoopIVs;
}

namespace {

template <typename ElementwiseNaryOp>
struct ScalarOp;

template <>
struct ScalarOp<ONNXAddOp> {
  using FOp = AddFOp;
  using IOp = AddIOp;
};

template <>
struct ScalarOp<ONNXMulOp> {
  using FOp = MulFOp;
  using IOp = MulIOp;
};

template <>
struct ScalarOp<ONNXDivOp> {
  using FOp = DivFOp;
  using IOp = SignedDivIOp;
};

template <>
struct ScalarOp<ONNXSubOp> {
  using FOp = SubFOp;
  using IOp = SubIOp;
};

template <>
struct ScalarOp<ONNXAndOp> {
  using FOp = AndOp; // not use
  using IOp = AndOp;
};

template <>
struct ScalarOp<ONNXOrOp> {
  using FOp = OrOp; // not use
  using IOp = OrOp;
};

template <>
struct ScalarOp<ONNXXorOp> {
  using FOp = XOrOp; // not use
  using IOp = XOrOp;
};

template <>
struct ScalarOp<ONNXExpOp> {
  using FOp = ExpOp;
  using IOp = ExpOp; // not use
};

template <>
struct ScalarOp<ONNXSumOp> {
  using FOp = AddFOp;
  using IOp = AddIOp;
};

template <>
struct ScalarOp<ONNXTanhOp> {
  using FOp = TanhOp;
  using IOp = TanhOp; // not use
};

template <>
struct ScalarOp<ONNXCosOp> {
  using FOp = CosOp;
  using IOp = CosOp; // not use
};

template <>
struct ScalarOp<ONNXLogOp> {
  using FOp = LogOp;
  using IOp = LogOp; // not use
};

template <>
struct ScalarOp<ONNXSqrtOp> {
  using FOp = KrnlSqrtOp;
  using IOp = KrnlSqrtOp; // not use
};

template <typename ElementwiseNaryOp>
using ScalarFOp = typename ScalarOp<ElementwiseNaryOp>::FOp;
template <typename ElementwiseNaryOp>
using ScalarIOp = typename ScalarOp<ElementwiseNaryOp>::IOp;

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename UnaryOp>
Value mapToLowerScalarOp(Operation *op, ArrayRef<Type> result_types,
                         ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) {
  /* Lower UnaryOp to Ops in the Standard dialect.
   */
  auto loc = op->getLoc();
  Type element_type = operands.front().getType();
  if (element_type.isa<IntegerType>()) {
    return rewriter.create<ScalarIOp<UnaryOp>>(loc, result_types, operands,
                                               mlir::None);
  } else if (element_type.isa<FloatType>()) {
    return rewriter.create<ScalarFOp<UnaryOp>>(loc, result_types, operands,
                                               mlir::None);
  } else {
    emitError(loc, "unsupported element type");
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSinhOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXSinhOp>(Operation *op, ArrayRef<Type> result_types,
                                     ArrayRef<Value> operands,
                                     ConversionPatternRewriter &rewriter) {
  // ONNXSinhOp(%X) = DivFOp(SubFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto two = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 2));
  auto neg = rewriter.create<SubFOp>(loc, zero, operand);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, rewriter.create<SubFOp>(loc, exp, negExp), two);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXCoshOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXCoshOp>(Operation *op, ArrayRef<Type> result_types,
                                     ArrayRef<Value> operands,
                                     ConversionPatternRewriter &rewriter) {
  // ONNXCoshOp(%X) = DivFOp(AddFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto two = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 2));
  auto neg = rewriter.create<SubFOp>(loc, zero, operand);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, rewriter.create<AddFOp>(loc, exp, negExp), two);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSigmoidOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXSigmoidOp>(Operation *op,
                                        ArrayRef<Type> result_types,
                                        ArrayRef<Value> operands,
                                        ConversionPatternRewriter &rewriter) {
  // ONNXSigmoidOp(%X) = DivFOp(ConstantOp 1,
  //                            AddFOp(ConstantOp 1, ExpOp(NegFOp(%X))))
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto one = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 1));
  auto neg = rewriter.create<SubFOp>(loc, zero, operand);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, one, rewriter.create<AddFOp>(loc, one, negExp));

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXHardSigmoidOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXHardSigmoidOp>(
    Operation *op, ArrayRef<Type> result_types, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) {
  // %Y = AddFOp(MulFOp(alpha, %X), beta)
  // %Z = SelectOp(CmpFOp(OGT, %Y, Constant 0),
  //               %Y,
  //               Constant 0)
  // ONNXHardSigmoidOp(%X) = SelectOp(CmpFOp(OLT, %Z, Constant 1),
  //                                  %Z,
  //                                  Constant 1)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXHardSigmoidOp>(op).alpha().convertToFloat());
  auto betaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXHardSigmoidOp>(op).beta().convertToFloat());
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto one = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 1));
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttribute);
  auto beta = rewriter.create<ConstantOp>(loc, betaAttribute);

  auto add = rewriter.create<AddFOp>(
      loc, rewriter.create<MulFOp>(loc, alpha, operand), beta);
  auto maxPredicate =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, add, zero);
  auto max = rewriter.create<SelectOp>(loc, maxPredicate, add, zero);
  auto minPredicate =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, max, one);
  auto result = rewriter.create<SelectOp>(loc, minPredicate, max, one);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXEluOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXEluOp>(Operation *op, ArrayRef<Type> result_types,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) {
  // ONNXEluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                          MulFOp(alpha, SubFOp(ExpOp(%X), 1)),
  //                          %X)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXEluOp>(op).alpha().convertToFloat());
  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto one = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 1));
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttribute);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto lessThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<SelectOp>(
      loc, lessThanZero,
      rewriter.create<MulFOp>(loc, alpha,
                              rewriter.create<SubFOp>(loc, exp, one)),
      operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReluOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXReluOp>(Operation *op, ArrayRef<Type> result_types,
                                     ArrayRef<Value> operands,
                                     ConversionPatternRewriter &rewriter) {
  // ONNXReluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                           ConstantOp 0,
  //                           %X)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto lessThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<SelectOp>(loc, lessThanZero, zero, operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXLeakyReluOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXLeakyReluOp>(Operation *op,
                                          ArrayRef<Type> result_types,
                                          ArrayRef<Value> operands,
                                          ConversionPatternRewriter &rewriter) {
  // ONNXLeakyReluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                                MulFOp(alpha, %X),
  //                                %X)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXLeakyReluOp>(op).alpha().convertToFloat());
  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttribute);
  auto lessThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<SelectOp>(
      loc, lessThanZero, rewriter.create<MulFOp>(loc, alpha, operand), operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSeluOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXSeluOp>(Operation *op, ArrayRef<Type> result_types,
                                     ArrayRef<Value> operands,
                                     ConversionPatternRewriter &rewriter) {
  // ONNXSeluOp(%X) = SelectOp(CmpFOp(OGT, %X, ConstantOp 0),
  //                           MulFOp(gamma, %X),
  //                           MulFOp(gamma,
  //                                  SubFOp(MulFOp(alpha, ExpOp(%X)),
  //                                         alpha)))
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXSeluOp>(op).alpha().convertToFloat());
  auto gammaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXSeluOp>(op).gamma().convertToFloat());
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttribute);
  auto gamma = rewriter.create<ConstantOp>(loc, gammaAttribute);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto greaterThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, operand, zero);
  auto select = rewriter.create<SelectOp>(
      loc, greaterThanZero, operand,
      rewriter.create<SubFOp>(loc, rewriter.create<MulFOp>(loc, alpha, exp),
                              alpha));
  auto result = rewriter.create<MulFOp>(loc, gamma, select);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReciprocalOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXReciprocalOp>(
    Operation *op, ArrayRef<Type> result_types, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) {
  // ONNXReciprocalOp(%X) = DivFOp(ConstantOp 1, %X)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto one = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 1));
  auto result = rewriter.create<DivFOp>(loc, one, operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSoftplusOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXSoftplusOp>(
    Operation *op, ArrayRef<Type> result_types, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) {
  // ONNXSoftplusOp(%X) = LogOp(AddFOp(ExpOp(%X), ConstantOp 1))
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto one = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 1));
  auto add = rewriter.create<AddFOp>(loc, exp, one);
  auto result = rewriter.create<LogOp>(loc, add);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSoftsignOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXSoftsignOp>(
    Operation *op, ArrayRef<Type> result_types, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) {
  // ONNXSoftsignOp(%X) = DivFOp(ConstantOp 1, %X)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto abs = rewriter.create<AbsFOp>(loc, operand);
  auto one = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 1));
  auto add = rewriter.create<AddFOp>(loc, abs, one);
  auto result = rewriter.create<DivFOp>(loc, operand, add);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSignOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXSignOp>(Operation *op, ArrayRef<Type> result_types,
                                     ArrayRef<Value> operands,
                                     ConversionPatternRewriter &rewriter) {

  auto loc = op->getLoc();
  Value operand = operands[0];
  Type element_type = operands.front().getType();
  // TODO: unsigned int should be supported separately?
  if (element_type.isa<IntegerType>()) {
    // %Y = SelectOP(CmpIOp(GT, %X, ConstantOp 0),
    //               ConstantOp 1,
    //               COnstantOp -1)
    // ONNXSignOp(%X) = SelectOP(CmpIOp(EQ, %X, ConstantOp 0),
    //                           ConstantOp 0,
    //                           %Y)
    auto zero = rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    auto one = rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
    auto minusOne =
        rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(-1));
    auto plusPredicate =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, operand, zero);
    auto plusSelect =
        rewriter.create<SelectOp>(loc, plusPredicate, one, minusOne);
    auto zeroPredicate =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, operand, zero);
    auto result =
        rewriter.create<SelectOp>(loc, zeroPredicate, zero, plusSelect);
    return result;
  } else if (element_type.isa<FloatType>()) {
    // %Y = SelectOP(CmpFOp(OGT, %X, ConstantOp 0),
    //               ConstantOp 1,
    //               ConstantOp -1)
    // ONNXSignOp(%X) = SelectOP(CmpFOp(OEQ, %X, ConstantOp 0),
    //                           ConstantOp 0,
    //                           %Y)
    auto zero =
        rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    auto one = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
    auto minusOne =
        rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(-1.0f));
    auto plusPredicate =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, operand, zero);
    auto plusSelect =
        rewriter.create<SelectOp>(loc, plusPredicate, one, minusOne);
    auto zeroPredicate =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, operand, zero);
    auto result =
        rewriter.create<SelectOp>(loc, zeroPredicate, zero, plusSelect);
    return result;
  } else {
    emitError(loc, "unsupported element type");
  }
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMaxOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXMaxOp>(Operation *op, ArrayRef<Type> result_types,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) {
  // ONNXMaxOp(%X, %Y) = SelectOp(CmpFOp(OGT, %X, %Y),
  //                              %X,
  //                              %Y)
  auto loc = op->getLoc();
  Value lhs = operands[0];
  Value rhs = operands[1];
  auto max = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
  auto result = rewriter.create<SelectOp>(loc, max, lhs, rhs);
  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMinOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXMinOp>(Operation *op, ArrayRef<Type> result_types,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) {
  // ONNXMinOp(%X, %Y) = SelectOp(CmpFOp(OLT, %X, %Y),
  //                              %X,
  //                              %Y)
  auto loc = op->getLoc();
  Value lhs = operands[0];
  Value rhs = operands[1];
  auto min = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
  auto result = rewriter.create<SelectOp>(loc, min, lhs, rhs);
  return result;
}

// Element-wise unary ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLowering : public ConversionPattern {
  ONNXElementwiseUnaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(ElementwiseUnaryOp::getOperationName(), 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO: Check that the types are valid.
    // An element-wise unary operation must have all operands and the result of
    // the same type. This should have been verified by the verifier.
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);

    // If the output has a dynamic dimension, pass the operands required for
    // each dynamic dimension to the AllocOp. The first operand of the
    // operation is used. The operands of the op need to match in terms of
    // dimensions with the result at this pre-optimization phase.
    // TODO: verify that dimensions match.
    // TODO: can the dimension of the result differ after optimizations?
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc,
                                    {operands[0]});

    std::vector<Value> originalLoops;
    KrnlOptimizeLoopsOp optimizedLoopsOp;
    KrnlIterateOp iterateOp;
    emitKrnlLoopsAndIterationForOperand(
        rewriter, loc, operands[0], originalLoops,
        optimizedLoopsOp, iterateOp);
    Block &optimizationBlock = optimizedLoopsOp.region().front();
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // 1. Insert any optimizations in the KrnlOptimizeLoopsOp body.
    rewriter.setInsertionPointToEnd(&optimizationBlock);
    // Return from KrnlOptimizeLoopsOp body.
    // When no optimizations are present we just return the loops
    // unchaged.
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);

    // 2. Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation:
    SmallVector<Value, 4> loopIVs;
    for (auto arg : iterationBlock.getArguments())
      loopIVs.push_back(arg);

    auto loadedVal = rewriter.create<LoadOp>(loc, operands[0], loopIVs);
    auto loweredOpResult = mapToLowerScalarOp<ElementwiseUnaryOp>(
        op, memRefType.getElementType(), {loadedVal}, rewriter);
    // Store result in the resulting array.
    rewriter.create<StoreOp>(loc, loweredOpResult, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

// Element-wise variadic ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLowering : public ConversionPattern {
  ONNXElementwiseVariadicOpLowering(MLIRContext *ctx)
      : ConversionPattern(ElementwiseVariadicOp::getOperationName(), 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO: Check that the types are valid.
    // An element-wise variadic operation must have all operands and the result
    // of the same type. This should have been verified by the verifier.
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();
    auto numArgs = op->getNumOperands();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    // If the output has a dynamic dimension, we compute its dimension at
    // runtime by using dimensions from the operands.
    // In particular, we need to know from which operand a result dimension
    // comes from.
    // TODO: can the dimension of the result differ after optimizations?
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc,
                                    operands);

    // Get run-time dimension information for unknown dimensions used for
    // broadcasting.
    std::map<int, std::map<int, Value>> broadcastedDimInfo =
        getBroadcastedDimInfo(loc, rewriter, memRefType, operands);

    std::vector<Value> originalLoops;
    KrnlOptimizeLoopsOp optimizedLoopsOp;
    KrnlIterateOp iterateOp;
    emitKrnlLoopsAndIterationForOperand(
        rewriter, loc, alloc, originalLoops,
        optimizedLoopsOp, iterateOp);
    Block &optimizationBlock = optimizedLoopsOp.region().front();
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // 1. Insert any optimizations in the KrnlOptimizeLoopsOp body.
    rewriter.setInsertionPointToEnd(&optimizationBlock);
    // Return from KrnlOptimizeLoopsOp body.
    // When no optimizations are present we just return the loops unchaged.
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);

    // 2. Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation:
    SmallVector<Value, 4> loopIVs;
    for (auto arg : iterationBlock.getArguments())
      loopIVs.push_back(arg);

    // Fold over operands for each of their scalar values
    Value accumulated, next;
    auto accumulatedLoopIVs = getLoopIVsForBroadcasting(
        loc, rewriter, loopIVs, operands[0], broadcastedDimInfo[0]);
    accumulated = rewriter.create<LoadOp>(loc, operands[0], accumulatedLoopIVs);
    for (unsigned i = 1; i < numArgs; i++) {
      auto nextLoopIVs = getLoopIVsForBroadcasting(
          loc, rewriter, loopIVs, operands[i], broadcastedDimInfo[i]);
      next = rewriter.create<LoadOp>(loc, operands[i], nextLoopIVs);
      accumulated = mapToLowerScalarOp<ElementwiseVariadicOp>(
          op, memRefType.getElementType(), {accumulated, next}, rewriter);
    }
    // Store result in the resulting array.
    rewriter.create<StoreOp>(loc, accumulated, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

struct ONNXSoftmaxOpLowering : public ConversionPattern {
  ONNXSoftmaxOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSoftmaxOp::getOperationName(), 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // softmax(x) = let max_x = max(x) in
    //                let exp_x = exp(x - max_x) in
    //                  let sum = sum(exp_x) in
    //                    exp_x / sum
    auto tensorType = (*op->result_type_begin()).cast<RankedTensorType>();
    int64_t rank = tensorType.getRank();
    int64_t axis = llvm::dyn_cast<ONNXSoftmaxOp>(op).axis().getSExtValue();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);

    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);
    auto elementType = memRefType.getElementType();

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc,
                                    operands[0]);

    // Shape of the result
    auto memRefShape = memRefType.getShape();

    // Insert allocations and deallocations for sum and max.
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    Value sumOp = insertAllocAndDealloc(scalarMemRefType, loc, rewriter, true);
    Value maxOp = insertAllocAndDealloc(scalarMemRefType, loc, rewriter, true);
    Value zero =
        rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
    Value negInfinity = rewriter.create<ConstantOp>(
        loc,
        FloatAttr::get(elementType, -std::numeric_limits<float>::infinity()));

    // Define loops.
    std::vector<Value> originalLoops;
    std::vector<Value> optimizedLoops;
    Block *optimizationBlock = defineLoops(rewriter, loc, originalLoops,
            optimizedLoops, rank);

    // Coerce the input into a 2-D tensor. `axis` will be the coercing point.
    // This coercing follows the softmax definition in ONNX:
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
    // Here, we create an outer loop and inner loop for handling the two
    // dimensions. The outer loop is only created once `axis` is not zero.

    // Define an outer loop with respect to axis.
    std::vector<Value> outerLoops, optimizedOuterLoops;
    outerLoops.reserve(axis);
    optimizedOuterLoops.reserve(axis);
    for (int i = 0; i < axis; ++i) {
      outerLoops.push_back(originalLoops[i]);
      optimizedOuterLoops.push_back(optimizedLoops[i]);
    }
    KrnlIterateOperandPack outerPack(rewriter, outerLoops, optimizedOuterLoops);
    for (int i = 0; i < axis; ++i)
      addDimensionToPack(rewriter, loc, outerPack, operands[0], i);

    // Define an inner loop with respect to axis.
    std::vector<Value> innerLoops, optimizedInnerLoops;
    innerLoops.reserve(rank - axis);
    optimizedInnerLoops.reserve(rank - axis);
    for (int i = axis; i < rank; ++i) {
      innerLoops.push_back(originalLoops[i]);
      optimizedInnerLoops.push_back(optimizedLoops[i]);
    }
    KrnlIterateOperandPack innerPack(rewriter, innerLoops, optimizedInnerLoops);
    for (int i = axis; i < rank; ++i)
      addDimensionToPack(rewriter, loc, innerPack, operands[0], i);

    KrnlIterateOp outerIterateOp, maxIterateOp, sumIterateOp, softmaxIterateOp;
    SmallVector<Value, 4> outerLoopIVs;
    if (axis != 0) {
      outerIterateOp = rewriter.create<KrnlIterateOp>(loc, outerPack);

      // No optimization
      rewriter.setInsertionPointToEnd(optimizationBlock);
      rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);

      // Insert instructions inside the outer loop.
      Block &outerIterationBlock = outerIterateOp.bodyRegion().front();
      rewriter.setInsertionPointToStart(&outerIterationBlock);
      for (auto arg : outerIterationBlock.getArguments())
        outerLoopIVs.push_back(arg);

      // Reset accumulators.
      rewriter.create<StoreOp>(loc, zero, sumOp);
      rewriter.create<StoreOp>(loc, negInfinity, maxOp);

      // Create an inner loop to compute max.
      maxIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);
      // Create an inner loop to compute sum.
      sumIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);
      // Create an inner loop to compute softmax.
      softmaxIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);
    } else {
      // Reset accumulators.
      rewriter.create<StoreOp>(loc, zero, sumOp);
      rewriter.create<StoreOp>(loc, negInfinity, maxOp);

      // Create an inner loop to compute max.
      maxIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);
      // Create an inner loop to compute sum.
      sumIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);
      // Create an inner loop to compute softmax.
      softmaxIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);

      // No optimization
      rewriter.setInsertionPointToEnd(optimizationBlock);
      rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);
    }

    // Insert instructions inside the max loop.
    Block &maxIterationBlock = maxIterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&maxIterationBlock);

    // Get induction variables.
    SmallVector<Value, 4> maxLoopIVs;
    for (auto arg : outerLoopIVs)
      maxLoopIVs.push_back(arg);
    for (auto arg : maxIterationBlock.getArguments())
      maxLoopIVs.push_back(arg);

    // Compute the max value.
    Value max = rewriter.create<LoadOp>(loc, maxOp);
    Value nextMax = rewriter.create<LoadOp>(loc, operands[0], maxLoopIVs);
    auto maxCond =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, max, nextMax);
    max = rewriter.create<SelectOp>(loc, maxCond, max, nextMax);
    rewriter.create<StoreOp>(loc, max, maxOp);

    // Get the max.
    rewriter.setInsertionPoint(sumIterateOp);
    max = rewriter.create<LoadOp>(loc, maxOp);

    // Insert instructions inside the sum loop.
    Block &sumIterationBlock = sumIterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&sumIterationBlock);

    // Get induction variables.
    SmallVector<Value, 4> sumLoopIVs;
    for (auto arg : outerLoopIVs)
      sumLoopIVs.push_back(arg);
    for (auto arg : sumIterationBlock.getArguments())
      sumLoopIVs.push_back(arg);

    // Sum up values.
    Value sum = rewriter.create<LoadOp>(loc, sumOp);
    Value next = rewriter.create<LoadOp>(loc, operands[0], sumLoopIVs);
    Value sub = rewriter.create<SubFOp>(loc, next, max);
    Value exp = rewriter.create<ExpOp>(loc, sub);
    sum = rewriter.create<AddFOp>(loc, sum, exp);
    rewriter.create<StoreOp>(loc, sum, sumOp);
    // Store intermediate values in the result to avoid recomputation.
    rewriter.create<StoreOp>(loc, exp, alloc, sumLoopIVs);

    // Get the sum.
    rewriter.setInsertionPoint(softmaxIterateOp);
    sum = rewriter.create<LoadOp>(loc, sumOp);

    // Insert instructions inside the softmax loop.
    Block &softmaxIterationBlock = softmaxIterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&softmaxIterationBlock);

    // Get induction variables.
    SmallVector<Value, 4> softmaxLoopIVs;
    for (auto arg : outerLoopIVs)
      softmaxLoopIVs.push_back(arg);
    for (auto arg : softmaxIterationBlock.getArguments())
      softmaxLoopIVs.push_back(arg);

    // Compute softmax.
    Value expLoadedVal = rewriter.create<LoadOp>(loc, alloc, softmaxLoopIVs);
    Value result = rewriter.create<DivFOp>(loc, expLoadedVal, sum);
    rewriter.create<StoreOp>(loc, result, alloc, softmaxLoopIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

struct ONNXReshapeOpLowering : public ConversionPattern {
  ONNXReshapeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXReshapeOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);
    Value alloc;

    // Compute size in bytes.
    Value tensorSize = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                     getMemRefEltSizeInBytes(memRefType)));
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    } else {
      auto memRefShape = memRefType.getShape();
      auto inputShape = operands[0].getType().cast<MemRefType>().getShape();
      SmallVector<Value, 4> allocOperands;
      for (int i = 0; i < memRefShape.size(); ++i) {
        // The shape array can always be used to construct shape information of
        // the result.
        Value index = rewriter.create<ConstantOp>(
            loc, rewriter.getIntegerAttr(rewriter.getIndexType(), i));
        // Load index from array of indices.
        Value loadedVal = rewriter.create<LoadOp>(loc, operands[1], index);
        // If a dimension is zero, the actual dimension value is taken from the
        // input tensor.
        if (i < inputShape.size()) {
          Value dimVal;
          auto dimTy = loadedVal.getType().cast<IntegerType>();
          if (inputShape[i] < 0) {
            Value dim = rewriter.create<DimOp>(loc, operands[0], i);
            dimVal = rewriter.create<IndexCastOp>(loc, dim, dimTy);
          } else {
            dimVal = rewriter.create<ConstantOp>(
                loc, rewriter.getIntegerAttr(dimTy, inputShape[i]));
          }
          auto zero = rewriter.create<ConstantOp>(
              loc, rewriter.getIntegerAttr(dimTy, 0));
          auto isZero =
              rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, loadedVal, zero);
          loadedVal = rewriter.create<SelectOp>(loc, isZero, dimVal, loadedVal);
        }
        // Check if the loaded index is already the correct width of 64 bits.
        // Convert the value to a 64 bit integer if needed.
        Value int64LoadedVal = loadedVal;
        if (loadedVal.getType().cast<IntegerType>().getWidth() < 64)
          int64LoadedVal = rewriter.create<ZeroExtendIOp>(
              loc, loadedVal, rewriter.getIntegerType(64));
        tensorSize = rewriter.create<MulIOp>(loc, tensorSize, int64LoadedVal);
        allocOperands.push_back(rewriter.create<IndexCastOp>(
              loc, loadedVal, rewriter.getIndexType()));
      }
      AllocOp allocateMemref =
          rewriter.create<AllocOp>(loc, memRefType, allocOperands);

      // Make sure to allocate at the beginning of the block if
      // all dimensions are known.
      auto *parentBlock = allocateMemref.getOperation()->getBlock();
      if (insertDealloc) {
        auto dealloc = rewriter.create<DeallocOp>(loc, allocateMemref);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }

      alloc = allocateMemref;
    }

    rewriter.create<KrnlMemcpyOp>(loc, alloc, operands[0], tensorSize);
    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

struct ONNXGemmOpLowering : public ConversionPattern {
  ONNXGemmOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXGemmOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    Value A, B, C;
    A = operands[0];
    B = operands[1];
    C = operands[2];

    auto alphaAttr = FloatAttr::get(tensorType.getElementType(),
        llvm::dyn_cast<ONNXGemmOp>(op).alpha().convertToFloat());
    auto betaAttr = FloatAttr::get(tensorType.getElementType(),
        llvm::dyn_cast<ONNXGemmOp>(op).beta().convertToFloat());
    auto alpha = rewriter.create<ConstantOp>(loc, alphaAttr);
    auto beta = rewriter.create<ConstantOp>(loc, betaAttr);

    bool isTransA = (llvm::dyn_cast<ONNXGemmOp>(op).transA() != 0);
    bool isTransB = (llvm::dyn_cast<ONNXGemmOp>(op).transB() != 0);

    // Result type
    auto memRefType = convertTensorToMemRef(tensorType);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else {
      auto memRefShape = memRefType.getShape();
      SmallVector<Value, 2> allocOperands;
      if (memRefShape[0] < 0) {
        auto dim = rewriter.create<DimOp>(loc, A, (isTransA) ? 1 : 0);
        allocOperands.emplace_back(dim);
      }
      if (memRefShape[1] < 0) {
        auto dim = rewriter.create<DimOp>(loc, B, (isTransB) ? 0 : 1);
        allocOperands.emplace_back(dim);
      }
      alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
      if (insertDealloc) {
        auto *parentBlock = alloc.getDefiningOp()->getBlock();
        auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t numLoops = 3;

    // Define loops.
    std::vector<Value> originalLoops;
    std::vector<Value> optimizedLoops;
    Block *optimizationBlock = defineLoops(rewriter, loc, originalLoops,
            optimizedLoops, numLoops);

    // We have two Krnl loops:
    // - Outer loop iterates over the output matrix dimensions, and
    // - Reduction loop iterates over the reduction dimension.

    // Outer loop
    std::vector<Value> outerLoops, optimizedOuterLoops;
    outerLoops.reserve(2);
    optimizedOuterLoops.reserve(2);
    for (int i = 0; i < 2; ++i) {
      outerLoops.push_back(originalLoops[i]);
      optimizedOuterLoops.push_back(optimizedLoops[i]);
    }
    KrnlIterateOperandPack outerPack(rewriter, outerLoops,
                                      optimizedOuterLoops);
    // Induction variables for the outer loops
    for (int i = 0; i < 2; ++i)
      addDimensionToPack(rewriter, loc, outerPack, alloc, i);

    // Reduction loop
    std::vector<Value> reductionLoops, optimizedReductionLoops;
    reductionLoops.reserve(1);
    optimizedReductionLoops.reserve(1);
    reductionLoops.push_back(originalLoops[2]);
    optimizedReductionLoops.push_back(optimizedLoops[2]);
    KrnlIterateOperandPack reductionPack(rewriter, reductionLoops,
                                         optimizedReductionLoops);
    // Induction variable for the reduction dimension
    // Try to find and use a static value from A or B first.
    // If it failed then use a dynamic value.
    auto ATy = A.getType().cast<MemRefType>();
    auto BTy = B.getType().cast<MemRefType>();
    int64_t K_A_Idx = (isTransA) ? 0 : 1;
    int64_t K_B_Idx = (isTransB) ? 1 : 0;
    reductionPack.pushConstantBound(0);
    if (ATy.getShape()[K_A_Idx] != -1)
        reductionPack.pushConstantBound(ATy.getShape()[K_A_Idx]);
    else
      if (BTy.getShape()[K_B_Idx] != -1)
        reductionPack.pushConstantBound(BTy.getShape()[K_B_Idx]);
      else
        reductionPack.pushOperandBound(
            rewriter.create<DimOp>(loc, B, K_B_Idx).getResult());

    // Get run-time dimension information for unknown dimensions used for
    // broadcasting.
    // GemmOp supports unidirectional broadcasting from C to A*B.
    // Hence, it must be enough to get broadcasting information for C only.
    std::map<int, Value> broadcastedDimInfo;
    auto shape = C.getType().cast<MemRefType>().getShape();
    for (int i = 0; i < shape.size(); ++i) {
      if (shape[i] < 0) {
        auto dim = rewriter.create<DimOp>(loc, C, i).getResult();
        auto one = rewriter.create<ConstantIndexOp>(loc, 1);
        auto isBroadcasted =
          rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, dim, one);
        broadcastedDimInfo.insert(std::make_pair(i, isBroadcasted));
      }
    }

    auto outerIterateOp = rewriter.create<KrnlIterateOp>(loc, outerPack);

    // Now perform the insertions into the body of the
    // just generated instructions:

    // No optimization
    rewriter.setInsertionPointToEnd(optimizationBlock);
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);

    // Insert instructions inside the outer loop.
    Block &outerIterationBlock = outerIterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&outerIterationBlock);

    // Induction variables
    SmallVector<Value, 4> loopMNIVs;
    for (auto arg : outerIterationBlock.getArguments()) {
      loopMNIVs.emplace_back(arg);
    }

    // Initialize the output of A*B
    auto zero = rewriter.create<ConstantOp>(
        loc, FloatAttr::get(memRefType.getElementType(), 0));
    rewriter.create<StoreOp>(loc, zero, alloc, loopMNIVs);

    // Compute A*B
    auto matmulIterateOp = rewriter.create<KrnlIterateOp>(loc, reductionPack);

    // Compute beta*C, and add up to alpha*A*B (unidirectional broadcasting)
    auto loopCIVs = getLoopIVsForBroadcasting(
        loc, rewriter, loopMNIVs, C, broadcastedDimInfo);
    auto loadedC = rewriter.create<LoadOp>(loc, C, loopCIVs);
    auto loadedAB = rewriter.create<LoadOp>(loc, alloc, loopMNIVs);
    auto alphaAB = rewriter.create<MulFOp>(loc, alpha, loadedAB);
    auto betaC = rewriter.create<MulFOp>(loc, beta, loadedC);
    auto Y = rewriter.create<AddFOp>(loc, alphaAB, betaC);
    rewriter.create<StoreOp>(loc, Y, alloc, loopMNIVs);

    // Insert instructions to do matrix multiplication: A*B
    Block &matmulIterationBlock = matmulIterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&matmulIterationBlock);

    // Induction variables
    SmallVector<Value, 4> loopKIVs, loopAIVs, loopBIVs;
    for (auto arg : matmulIterationBlock.getArguments())
      loopKIVs.emplace_back(arg);
    if (isTransA) {
      loopAIVs.emplace_back(loopKIVs[0]);
      loopAIVs.emplace_back(loopMNIVs[0]);
    } else {
      loopAIVs.emplace_back(loopMNIVs[0]);
      loopAIVs.emplace_back(loopKIVs[0]);
    }
    if (isTransB) {
      loopBIVs.emplace_back(loopMNIVs[1]);
      loopBIVs.emplace_back(loopKIVs[0]);
    } else {
      loopBIVs.emplace_back(loopKIVs[0]);
      loopBIVs.emplace_back(loopMNIVs[1]);
    }

    // Matmul computation
    auto loadedA = rewriter.create<LoadOp>(loc, A, loopAIVs);
    auto loadedB = rewriter.create<LoadOp>(loc, B, loopBIVs);
    auto loadedY = rewriter.create<LoadOp>(loc, alloc, loopMNIVs);
    auto AB = rewriter.create<MulFOp>(loc, loadedA, loadedB);
    auto accumulated = rewriter.create<AddFOp>(loc, loadedY, AB);
    rewriter.create<StoreOp>(loc, accumulated, alloc, loopMNIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

struct ONNXUnsqueezeOpLowering : public ConversionPattern {
  ONNXUnsqueezeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXUnsqueezeOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    int outRank = tensorType.getRank();

    // Assume that `axes` has been validated by shape inference.
    // So, here we just get it.
    ArrayAttr axisAttrs = llvm::dyn_cast<ONNXUnsqueezeOp>(op).axesAttr();
    SmallVector<int, 4> axes;
    for (auto axisAttr : axisAttrs.getValue()) {
      int axis = axisAttr.cast<IntegerAttr>().getInt();
      axis = axis >= 0 ? axis : (outRank + axis);
      axes.emplace_back(axis);
    }

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);
    Value alloc;

    // Compute size in bytes.
    Value tensorSize = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                     getMemRefEltSizeInBytes(memRefType)));

    bool insertDealloc = checkInsertDealloc(op);
    auto memRefShape = memRefType.getShape();
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
      for (int i = 0; i < memRefShape.size(); ++i) {
        Value dimVal = rewriter.create<ConstantOp>(
            loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                         memRefShape[i]));
        tensorSize = rewriter.create<MulIOp>(loc, tensorSize, dimVal);
      }
    } else {
      // Unknown dimensions are always the operand's dimensions.
      SmallVector<Value, 4> allocOperands;
      for (int outIdx = 0, inIdx = 0; outIdx < memRefShape.size(); ++outIdx) {
        Value dimVal = nullptr;
        if (memRefShape[outIdx] < 0) {
          Value index = rewriter.create<DimOp>(loc, operands[0], inIdx);
          dimVal = rewriter.create<IndexCastOp>(
              loc, index, rewriter.getIntegerType(64));
          allocOperands.emplace_back(index);
        } else {
          dimVal = rewriter.create<ConstantOp>(
              loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                           memRefShape[outIdx]));
        }
        tensorSize = rewriter.create<MulIOp>(loc, tensorSize, dimVal);
        if (std::find(axes.begin(), axes.end(), outIdx) == axes.end())
          inIdx++;
      }
      alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
      auto *parentBlock = alloc.getDefiningOp()->getBlock();
      if (insertDealloc) {
        auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }
    rewriter.create<KrnlMemcpyOp>(loc, alloc, operands[0], tensorSize);
    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};

struct ONNXTransposeOpLowering : public ConversionPattern {
  ONNXTransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTransposeOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();
    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc,
                                    {operands[0]});

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t rank = memRefShape.size();

    // Define loops.
    std::vector<Value> originalLoops;
    std::vector<Value> optimizedLoops;
    Block *optimizationBlock = defineLoops(rewriter, loc, originalLoops,
        optimizedLoops, rank);

    KrnlIterateOperandPack pack(rewriter, originalLoops, optimizedLoops);
    // Iterate over the loop nest using the input shape.
    for (int i = 0; i < rank; ++i)
      addDimensionToPack(rewriter, loc, pack, operands[0], i);

    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the
    // just generated instructions:

    // 1. Insert any optimizations in the KrnlOptimizeLoopsOp body.
    rewriter.setInsertionPointToEnd(optimizationBlock);
    // Return from KrnlOptimizeLoopsOp body.
    // When no optimizations are present we just return the loops
    // unchaged.
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);

    // 2. Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation.

    // Read perm attribute.
    SmallVector<int, 4> perm;
    auto permAttribute = llvm::dyn_cast<ONNXTransposeOp>(op).permAttr();
    if (permAttribute) {
      for (auto permVal : permAttribute.getValue())
        perm.emplace_back(permVal.cast<IntegerAttr>().getInt());
    } else {
      // TODO: Remove when perm is guaranteed to be present (even for
      // the default case). This means that perm was added by shape
      // inference or another pass to contain the values corresponding
      // to the default behavior of Transpose. 
      for (int i = iterationBlock.getArguments().size()-1; i >= 0; i--)
        perm.emplace_back(i);
    }

    SmallVector<Value, 4> inLoopIVs;
    for (auto arg : iterationBlock.getArguments())
      inLoopIVs.emplace_back(arg);

    SmallVector<Value, 4> outLoopIVs;
    for (int i=0; i<iterationBlock.getArguments().size(); ++i)
      outLoopIVs.emplace_back(iterationBlock.getArguments()[perm[i]]);

    auto inVal = rewriter.create<LoadOp>(loc, operands[0], inLoopIVs);
    rewriter.create<StoreOp>(loc, inVal, alloc, outLoopIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

struct ONNXIdentityOpLowering : public ConversionPattern {
  ONNXIdentityOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXIdentityOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, operands[0]);
    return matchSuccess();
  }
};

struct ONNXConvNoBiasOpLowering : public ConversionPattern {
  ONNXConvNoBiasOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConvNoBiasOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();
    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    ONNXConvNoBiasOp convOp = llvm::dyn_cast<ONNXConvNoBiasOp>(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc,
                                    {operands[0]});

    auto resultShape = memRefType.getShape();
    auto inputShape = operands[0].getType().cast<MemRefType>().getShape();
    auto kernelShape = operands[1].getType().cast<MemRefType>().getShape();

    // R = ConvNoBias(D, K)
    //
    // The input/output shapes will look like this:
    //
    // D (NxCxHxW) x K (MxC/groupxKHxKW) -> R (NxMxRHxRW)
    //
    // M is a multiple of the number of groups:
    //   M = group * kernelsPerGroup
    //
    // The loop nest will look as follows:
    //
    // kernelsPerGroup = M / group;
    // for n = 0 .. N:
    //   for g = 0 .. group:
    //     for m = 0 .. kernelsPerGroup:
    //       kernel = g * kernelsPerGroup + m;
    //       for r1 = 0 .. RH:
    //         for r2 = 0 .. RW:
    //           R[n][kernel][r1][r2] = 0;
    //           for c = 0 .. C/group:
    //             for k1 = 0 .. KH:
    //               for k2 = 0 .. KW:
    //                 R[n][kernel][r1][r2] =
    //                   D[n][g * (C / group) + c][r1 + k1][r2 + k2] *
    //                   K[kernel][c][k1][k2];
    //
    // TODO: handle padding.
    //
    // In the general case:
    //
    // D (NxCxD1xD2x...xDdim) x K (MxC/groupxK1xK2x...xKdim)
    //     -> R (NxMxR1xR2x...xRdim)
    //
    // The above loop nest can be adapted by increasing the number
    // of r- and k-index loop i.e. r1 r2 and k1 k2 loops.

    // Set up outermost loops: n g m r1 r2 ... rdim
    // Skip g if group is 1.

    // Before we start the iteration we need to compute the number of
    // unsplit kernels and fetch the number of groups from the attribute
    // list. Group is always a compilation constant.
    int64_t group = convOp.group().getSExtValue();
    // Compute the number of unsplit kernels. The number of kernels
    // must be a multiple of the number of groups.
    int64_t kernelsPerGroup = floor(kernelShape[0] / group);
    auto kernelsPerGroupValue =
        rewriter.create<ConstantIndexOp>(loc, kernelsPerGroup);
    auto zero = rewriter.create<ConstantOp>(
        loc, FloatAttr::get(memRefType.getElementType(), 0));
    Value subchannels;
    if (kernelShape[1] < 0) {
      subchannels =
          rewriter.create<DimOp>(loc, operands[1], 1).getResult();
    } else {
      subchannels = rewriter.create<ConstantIndexOp>(
          loc, kernelShape[1]);
    }

    // 1. Define outer loops and emit empty optimization block:
    int64_t nOuterLoops = (group > 1) ? 3 : 2;
    std::vector<Value> outerLoops;
    std::vector<Value> optimizedOuterLoops;
    Block *optimizationBlock = defineLoops(rewriter, loc, outerLoops,
        optimizedOuterLoops, nOuterLoops);

    // Prepare iteration arguments over outer loop nest.
    KrnlIterateOperandPack pack(
        rewriter, outerLoops, optimizedOuterLoops);
    //   for n = 0 .. N:
    pack.pushConstantBound(0);
    if (inputShape[0] < 0)
      pack.pushOperandBound(
          rewriter.create<DimOp>(loc, operands[0], 0).getResult());
    else
      pack.pushConstantBound(inputShape[0]);
    //   for g = 0 .. N:
    if (group > 1) {
      pack.pushConstantBound(0);
      pack.pushConstantBound(group);
    }
    //   for m = 0 .. kernelsPerGroup:
    pack.pushConstantBound(0);
    pack.pushConstantBound(kernelsPerGroup);
    // Outer loop iteration.
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &outerIterationBlock = iterateOp.bodyRegion().front();
    // Emit optimizations for outer loops:
    rewriter.setInsertionPointToEnd(optimizationBlock);
    rewriter.create<KrnlReturnLoopsOp>(loc, outerLoops);
    rewriter.setInsertionPointToStart(&outerIterationBlock);
    {
      // 2. Emit the body of the outer loop nest.

      // 2.1 Compute kernel order number: kernel = g * kernelsPerGroup + m;
      // If group is not set then the value of the kernel ID is
      // identical to that of the loop over kernels.
      Value kernel = outerIterationBlock.getArguments()[1];
      if (group > 1) {
        // Middle loop is over groups and third loop is over the
        // kernel identifiers in the current group.
        auto kernelsOffset = rewriter.create<MulIOp>(loc,
            outerIterationBlock.getArguments()[1],
            kernelsPerGroupValue);
        kernel = rewriter.create<AddIOp>(loc, kernelsOffset,
            outerIterationBlock.getArguments()[2]);
      }

      // 2.2 Define spatial loops
      int64_t nSpatialLoops = resultShape.size() - 2;
      std::vector<Value> spatialLoops;
      std::vector<Value> optimizedSpatialLoops;
      Block *optSpatialLoopBlock = defineLoops(rewriter, loc, spatialLoops,
        optimizedSpatialLoops, nSpatialLoops);

      // 2.3 Prepare iteration arguments for spatial loop nest.
      KrnlIterateOperandPack spatialPack(
        rewriter, spatialLoops, optimizedSpatialLoops);
      for (int i = 2; i < resultShape.size(); ++i)
        addDimensionToPack(rewriter, loc, spatialPack, alloc, i);

      // 2.4 Emit loop nest over output spatial dimensions.
      //   for rX = 0 .. RX
      auto spatialIterateOp =
          rewriter.create<KrnlIterateOp>(loc, spatialPack);
      Block &spatialIterationBlock = spatialIterateOp.bodyRegion().front();
      // 2.5 Emit optimizations for outer loops:
      rewriter.setInsertionPointToEnd(optSpatialLoopBlock);
      rewriter.create<KrnlReturnLoopsOp>(loc, spatialLoops);
      rewriter.setInsertionPointToStart(&spatialIterationBlock);
      {
        // 3. Emit the body of the spatial loop nest.
        // 3.1 Emit: R[n][kernel][r1][r2] = 0;
        SmallVector<Value, 4> resultIndices;
        // n
        resultIndices.emplace_back(outerIterationBlock.getArguments()[0]);
        // kernel
        resultIndices.emplace_back(kernel);
        // rX
        for (auto arg : spatialIterationBlock.getArguments())
          resultIndices.emplace_back(arg);
        // Store initializer value into output location.
        rewriter.create<StoreOp>(loc, zero, alloc, resultIndices);

        // 3.2 Define inner loops.
        int64_t nInnerLoops = 1 + (kernelShape.size() - 2);
        std::vector<Value> innerLoops;
        std::vector<Value> optimizedInnerLoops;
        Block *optInnerLoopBlock = defineLoops(rewriter, loc, innerLoops,
            optimizedInnerLoops, nInnerLoops);

        // 3.3 Prepare iteration arguments for inner loop nest.
        KrnlIterateOperandPack innerPack(
            rewriter, innerLoops, optimizedInnerLoops);
        //   for c = 0 .. C/group
        innerPack.pushConstantBound(0);
        innerPack.pushConstantBound(kernelShape[1]);
        //   for Kx = 0 .. KX
        for (int i = 2; i < kernelShape.size(); ++i)
          addDimensionToPack(rewriter, loc, innerPack, operands[1], i);

        // 3.4 Emit inner loop nest.
        auto innerIterateOp =
            rewriter.create<KrnlIterateOp>(loc, innerPack);
        Block &innerIterationBlock = innerIterateOp.bodyRegion().front();
        // 3.5 Emit optimizations for outer loops:
        rewriter.setInsertionPointToEnd(optInnerLoopBlock);
        rewriter.create<KrnlReturnLoopsOp>(loc, innerLoops);
        rewriter.setInsertionPointToStart(&innerIterationBlock);
        {
          // 4. Emit inner loop body
          // R[n][kernel][r1][r2] =
          //   D[n][g * (C / group) + c][r1 + k1][r2 + k2] *
          //   K[kernel][c][k1][k2];

          // 4.1 Prepare indices for accesing the data tensor.
          SmallVector<Value, 4> dataIndices;
          // n
          dataIndices.emplace_back(outerIterationBlock.getArguments()[0]);
          // g * (C / group) + c
          Value channelDepth = innerIterationBlock.getArguments()[0];
          if (group > 1)
            channelDepth = rewriter.create<AddIOp>(loc, channelDepth,
                rewriter.create<MulIOp>(loc, subchannels,
                    outerIterationBlock.getArguments()[1]));
          dataIndices.emplace_back(channelDepth);
          // rX + kX
          for (int i = 0; i < kernelShape.size() - 2; ++i)
            dataIndices.emplace_back(
                rewriter.create<AddIOp>(loc,
                    spatialIterationBlock.getArguments()[i],
                    innerIterationBlock.getArguments()[i+1]));

          // 4.2 Prepare indices for accessing the kernel tensor.
          SmallVector<Value, 4> kernelIndices;
          // kernel
          kernelIndices.emplace_back(kernel);
          // c
          kernelIndices.emplace_back(innerIterationBlock.getArguments()[0]);
          // kX
          for (int i = 0; i < kernelShape.size() - 2; ++i)
            kernelIndices.emplace_back(
                innerIterationBlock.getArguments()[i+1]);

          // 4.3 Compute convolution.
          auto loadData =
              rewriter.create<LoadOp>(loc, operands[0], dataIndices);
          auto loadKernel =
              rewriter.create<LoadOp>(loc, operands[1], kernelIndices);
          auto loadPartialSum =
              rewriter.create<LoadOp>(loc, alloc, resultIndices);
          Value result = rewriter.create<AddFOp>(loc, loadPartialSum,
              rewriter.create<MulFOp>(loc, loadData, loadKernel));
          // 4.4 Store computed value into output location.
          rewriter.create<StoreOp>(loc, result, alloc, resultIndices);
        }
      }
    }
    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// EntryPoint Op lowering to Krnl Entry Point.
//===----------------------------------------------------------------------===//

class ONNXEntryPointLowering : public OpRewritePattern<ONNXEntryPointOp> {
public:
  using OpRewritePattern<ONNXEntryPointOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ONNXEntryPointOp op,
                                     PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<KrnlEntryPointOp>(
        op,
        op.getAttrOfType<SymbolRefAttr>(
            ONNXEntryPointOp::getEntryPointFuncAttrName()),
        op.getAttrOfType<IntegerAttr>(ONNXEntryPointOp::getNumInputsAttrName()),
        op.getAttrOfType<IntegerAttr>(
            ONNXEntryPointOp::getNumOutputsAttrName()));
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// Conversion from Tensor type to the Standard dialect MemRef type.
//===----------------------------------------------------------------------===//

struct TensorTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) override {
    if (auto tensor_type = t.dyn_cast<TensorType>()) {
      results.push_back(convertTensorToMemRef(tensor_type));
      return success();
    }

    results.push_back(t);
    return success();
  }

  /// Return true if the inputs and outputs of the given function type are
  /// legal. [Taken from MLIR and adapted to only check the legality of the
  /// inputs. Once unranked results can be handled gracefully this
  /// override needs to be removed in favour of the original MLIR one.]
  bool isSignatureLegal(FunctionType funcType) {
    return llvm::all_of(funcType.getInputs(),
                        [this](Type type) { return isLegal(type); });
  }
};

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// Frontend to Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
namespace {
struct FrontendToKrnlLoweringPass
    : public ModulePass<FrontendToKrnlLoweringPass> {
  void runOnModule() final;
};
} // end anonymous namespace.

void FrontendToKrnlLoweringPass::runOnModule() {
  auto module = getModule();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target
      .addLegalDialect<KrnlOpsDialect, AffineOpsDialect, StandardOpsDialect>();

  // TODO: enable this once more ops are supported.
  // We also define the ONNX dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted.
  // target.addIllegalDialect<mlir::ONNXOpsDialect>();

  // TODO: add any other ops which are considered legal.
  // Some operations can be marked as being still legal.
  // Example: target.addLegalOp<mlir::OpName>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  OwningRewritePatternList patterns;

  // Convert TensorType to MemRef
  TensorTypeConverter tensor_to_memref_converter;
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    // FuncOp is legal only if types have been converted to Std types.
    return tensor_to_memref_converter.isSignatureLegal(op.getType());
  });

  // Type conversion for function signatures.
  // Call MLIR FuncOp signature conversion when result type is
  // a ranked tensor.
  populateFuncOpTypeConversionPattern(patterns, &getContext(),
                                      tensor_to_memref_converter);

  // Frontent operation lowering.
  patterns.insert<ONNXElementwiseUnaryOpLowering<mlir::ONNXExpOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXTanhOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXSinhOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXCoshOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXCosOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXLogOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXSigmoidOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXHardSigmoidOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXEluOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXReluOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXLeakyReluOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXSeluOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXReciprocalOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXSoftplusOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXSoftsignOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXSqrtOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXSignOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXAddOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXMulOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXDivOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXSubOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXAndOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXOrOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXXorOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXSumOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXMaxOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXMinOp>,
                  ONNXReshapeOpLowering, ONNXEntryPointLowering,
                  ONNXSoftmaxOpLowering, ONNXGemmOpLowering,
                  ONNXUnsqueezeOpLowering, ONNXTransposeOpLowering,
                  ONNXIdentityOpLowering, ONNXConvNoBiasOpLowering
                  >(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createLowerToKrnlPass() {
  return std::make_unique<FrontendToKrnlLoweringPass>();
}

static PassRegistration<FrontendToKrnlLoweringPass>
    pass("lower-frontend", "Lower frontend ops to Krnl dialect.");
