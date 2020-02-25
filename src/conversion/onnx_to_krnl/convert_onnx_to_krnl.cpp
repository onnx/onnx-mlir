//====- convert_onnx_to_krnl.cpp - ONNX dialects to Krnl lowering ---------===//
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

/// Get the corresponding MemRefType of a given TensorType/MemRefType.
static MemRefType convertToMemRefType(Type type) {
  MemRefType memRefType;
  auto tensorType = type.dyn_cast<TensorType>();
  if (tensorType) {
    assert(tensorType.hasRank() && "expected only ranked shapes");
    memRefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  } else {
    memRefType = type.dyn_cast<MemRefType>();
  }
  return memRefType;
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

// Create a mapping from result type's dimensions to input type's dimensions,
// given that the result type is the result of a reduction op over the input
// type.
std::map<int64_t, int64_t>
getReductionMapping(MemRefType inputTy, ArrayRef<int64_t> axes, bool keepdims) {
  std::map<int64_t, int64_t> OutInDimMap;
  int64_t rank = inputTy.getRank();

  // Mark reduction axes.
  std::vector<bool> isReductionAxis;
  for (decltype(rank) i = 0; i < rank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      isReductionAxis.push_back(true);
    else
      isReductionAxis.push_back(false);
  }

  for (decltype(rank) inIndex = 0, outIndex = 0; inIndex < rank; ++inIndex) {
    // If it is a reduction axis, there is no relationship among dimensions.
    if (isReductionAxis[inIndex]) {
      if (keepdims)
        outIndex++;
    } else {
      OutInDimMap.insert(std::make_pair(outIndex, inIndex));
      outIndex++;
    }
  }

  return OutInDimMap;
}

// Add bounds associated with the op operand to the KRNL iteration pack.
// Dynamic dimenions are supported.
static void addDimensionToPack(ConversionPatternRewriter &rewriter,
                               Location loc, KrnlIterateOperandPack &pack,
                               Value operand, int index) {
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
static KrnlOptimizeLoopsOp
emitOptimizedLoops(ConversionPatternRewriter &rewriter, Location loc,
                   std::vector<Value> &loops,
                   std::vector<Value> &optimizedLoops, int64_t numLoops) {
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
static Block *defineLoops(ConversionPatternRewriter &rewriter, Location loc,
                          std::vector<Value> &loops,
                          std::vector<Value> &optimizedLoops,
                          int64_t numLoops) {
  KrnlOptimizeLoopsOp optimizedLoopsOp =
      emitOptimizedLoops(rewriter, loc, loops, optimizedLoops, numLoops);
  return &optimizedLoopsOp.region().front();
}

// Function which emits a basic set of loops and optimized loops
// for a given operation argument. A reference to the loop optimization
// block is returned in the last argument of the function.
static void emitKrnlLoopsAndIterationForOperand(
    ConversionPatternRewriter &rewriter, Location loc, Value operand,
    std::vector<Value> &originalLoops, KrnlOptimizeLoopsOp &optimizedLoopsOp,
    KrnlIterateOp &iterateOp) {
  // Operand shape.
  auto shape = operand.getType().cast<MemRefType>().getShape();

  // Number of loops.
  int64_t rank = shape.size();

  // Define loops and optimized loops.
  std::vector<Value> optimizedLoops;
  optimizedLoopsOp =
      emitOptimizedLoops(rewriter, loc, originalLoops, optimizedLoops, rank);

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

// This is to get a scalar operation of a given type for a specific operation.
template <typename Op>
struct ScalarOp {
  using FOp = void;
  using IOp = void;
};

template <typename FOp>
using ScalarFOp = typename ScalarOp<FOp>::FOp;
template <typename IOp>
using ScalarIOp = typename ScalarOp<IOp>::IOp;

// Get the identity element of a operation.
// Return NULL if the function does not have identity.
template <typename DataType, typename Op>
DataType getIdentityValue() {
  return NULL;
}

//===----------------------------------------------------------------------===//
// This is used in the innermost loop of a KrnlIterateOp to insert computation
// composed of one or many scalar ops.
// Use template specialization for each of different ONNX operations.
//===----------------------------------------------------------------------===//
template <typename Op>
Value mapToLowerScalarOp(Operation *op, ArrayRef<Type> result_types,
                         ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  Type element_type = operands.front().getType();
  if (element_type.isa<IntegerType>()) {
    return rewriter.create<ScalarIOp<Op>>(loc, result_types, operands,
                                          mlir::None);
  } else if (element_type.isa<FloatType>()) {
    return rewriter.create<ScalarFOp<Op>>(loc, result_types, operands,
                                          mlir::None);
  } else {
    emitError(loc, "unsupported element type");
    return nullptr;
  }
}

// We divide the operator lowering into different categories.
// These categories are mostly similar to the operator categories in ONNX:
// https://github.com/onnx/onnx/tree/master/onnx/defs.
// Besides, it is better to put operators with the same computation pattern into
// the same category, e.g. element-wise operators will belong to the elementwise
// category.

// Math
#include "src/conversion/onnx_to_krnl/rewrite_patterns/math/elementwise.inc"
#include "src/conversion/onnx_to_krnl/rewrite_patterns/math/gemm.inc"
#include "src/conversion/onnx_to_krnl/rewrite_patterns/math/reduction.inc"
#include "src/conversion/onnx_to_krnl/rewrite_patterns/math/softmax.inc"
#include "src/conversion/onnx_to_krnl/rewrite_patterns/math/matmul.inc"
// Tensor
#include "src/conversion/onnx_to_krnl/rewrite_patterns/tensor/identity.inc"
#include "src/conversion/onnx_to_krnl/rewrite_patterns/tensor/reshape.inc"
#include "src/conversion/onnx_to_krnl/rewrite_patterns/tensor/transpose.inc"
#include "src/conversion/onnx_to_krnl/rewrite_patterns/tensor/unsqueeze.inc"
// Neural network
#include "src/conversion/onnx_to_krnl/rewrite_patterns/nn/conv.inc"
#include "src/conversion/onnx_to_krnl/rewrite_patterns/nn/normalization.inc"
#include "src/conversion/onnx_to_krnl/rewrite_patterns/nn/pooling.inc"

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

  TensorTypeConverter() {
    addConversion(convertType);
  }

  static LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) {
    if (auto type = convertToMemRefType(t)) {
      results.push_back(type);
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

  // Frontend operation lowering.
  // Math
  populateLoweringONNXElementwiseOpPattern(patterns, &getContext());
  populateLoweringONNXGemmOpPattern(patterns, &getContext());
  populateLoweringONNXReductionOpPattern(patterns, &getContext());
  populateLoweringONNXSoftmaxOpPattern(patterns, &getContext());
  populateLoweringONNXMatMulOpPattern(patterns, &getContext());
  // Tensor
  populateLoweringONNXReshapeOpPattern(patterns, &getContext());
  populateLoweringONNXUnsqueezeOpPattern(patterns, &getContext());
  populateLoweringONNXTransposeOpPattern(patterns, &getContext());
  populateLoweringONNXIdentityOpPattern(patterns, &getContext());
  // Neural network
  populateLoweringONNXConvOpPattern(patterns, &getContext());
  populateLoweringONNXNormalizationOpPattern(patterns, &getContext());
  populateLoweringONNXPoolingOpPattern(patterns, &getContext());
  // Entry point
  patterns.insert<ONNXEntryPointLowering>(&getContext());

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
