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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/compiler/dialect/krnl/krnl_helper.hpp"
#include "src/compiler/dialect/krnl/krnl_ops.hpp"
#include "src/compiler/dialect/onnx/onnx_ops.hpp"

#include "src/compiler/pass/passes.hpp"

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
static Value* insertAllocAndDealloc(MemRefType type, Location loc,
    PatternRewriter& rewriter, bool insertDealloc, Value* oldMemRef = nullptr) {
  // Put together alloc operands for any dynamic dimensions of the memref.
  AllocOp alloc;
  if (oldMemRef) {
    SmallVector<Value*, 4> allocOperands;
    auto memRefShape = type.getShape();
    for (int i = 0; i < memRefShape.size(); ++i)
      if (memRefShape[i] < 0)
        allocOperands.push_back(rewriter.create<DimOp>(loc, oldMemRef, i));
    alloc = rewriter.create<AllocOp>(loc, type, allocOperands);
  } else {
    alloc = rewriter.create<AllocOp>(loc, type);
  }

  // Make sure to allocate at the beginning of the block if
  // all dimensions are known.
  auto* parentBlock = alloc.getOperation()->getBlock();
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
static bool checkInsertDealloc(Operation* currentOp) {
  auto parentBlock = currentOp->getBlock();

  bool insertDealloc = true;
  parentBlock->walk([&insertDealloc, currentOp](ReturnOp op) {
    assert(currentOp->getNumResults() < 2 &&
        "No more than one result supported (for now).");
    // If there is at least one result to investigate.
    if (currentOp->getNumResults() > 0) {
      auto result = currentOp->getResult(0);
      for (auto operand : op.getOperands())
        if (operand == result)
          insertDealloc = false;
    }
  });

  return insertDealloc;
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
  using IOp = DivISOp;
};

template <>
struct ScalarOp<ONNXSubOp> {
  using FOp = SubFOp;
  using IOp = SubIOp;
};

template <>
struct ScalarOp<ONNXAndOp> {
  using FOp = AndOp;  // not use
  using IOp = AndOp;
};

template <>
struct ScalarOp<ONNXOrOp> {
  using FOp = OrOp;  // not use
  using IOp = OrOp;
};

template <>
struct ScalarOp<ONNXXorOp> {
  using FOp = XOrOp;  // not use
  using IOp = XOrOp;
};

template <>
struct ScalarOp<ONNXExpOp> {
  using FOp = ExpOp;
  using IOp = ExpOp;  // not use
};

template <typename ElementwiseNaryOp>
using ScalarFOp = typename ScalarOp<ElementwiseNaryOp>::FOp;
template <typename ElementwiseNaryOp>
using ScalarIOp = typename ScalarOp<ElementwiseNaryOp>::IOp;

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename UnaryOp>
Value* mapToLowerScalarOp(Location loc, ArrayRef<Type> result_types,
    ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) {
  /* Lower UnaryOp to Ops in the Standard dialect.
   */

  Type element_type = operands.front()->getType();
  if (element_type.isa<IntegerType>()) {
    return rewriter.create<ScalarIOp<UnaryOp>>(
        loc, result_types, operands, mlir::None);
  } else if (element_type.isa<FloatType>()) {
    return rewriter.create<ScalarFOp<UnaryOp>>(
        loc, result_types, operands, mlir::None);
  } else {
    emitError(loc, "unsupported element type");
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXTanhOp
//===----------------------------------------------------------------------===//
template <>
Value* mapToLowerScalarOp<ONNXTanhOp>(Location loc, ArrayRef<Type> result_types,
    ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) {
  // ONNXTanhOp(%X) = DivFOp(SubFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         AddFOp(ExpOp(%X), ExpOp(NegFOp(%X))))
  Value* operand = operands[0];
  auto zero = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  auto neg = rewriter.create<SubFOp>(loc, zero, operand);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result =
      rewriter.create<DivFOp>(loc, rewriter.create<SubFOp>(loc, exp, negExp),
          rewriter.create<AddFOp>(loc, exp, negExp));
  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSinhOp
//===----------------------------------------------------------------------===//
template <>
Value* mapToLowerScalarOp<ONNXSinhOp>(Location loc, ArrayRef<Type> result_types,
    ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) {
  // ONNXSinhOp(%X) = DivFOp(SubFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  Value* operand = operands[0];
  auto zero = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  auto two = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(2.0f));
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
Value* mapToLowerScalarOp<ONNXCoshOp>(Location loc, ArrayRef<Type> result_types,
    ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) {
  // ONNXCoshOp(%X) = DivFOp(AddFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  Value* operand = operands[0];
  auto zero = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  auto two = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(2.0f));
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
Value* mapToLowerScalarOp<ONNXSigmoidOp>(Location loc,
    ArrayRef<Type> result_types, ArrayRef<Value*> operands,
    ConversionPatternRewriter& rewriter) {
  // ONNXSigmoidOp(%X) = DivFOp(ConstantOp 1,
  //                            AddFOp(ConstantOp 1, ExpOp(NegFOp(%X))))
  Value* operand = operands[0];
  auto zero = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  auto one = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
  auto neg = rewriter.create<SubFOp>(loc, zero, operand);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, one, rewriter.create<AddFOp>(loc, one, negExp));
  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReluOp
//===----------------------------------------------------------------------===//
template <>
Value* mapToLowerScalarOp<ONNXReluOp>(Location loc, ArrayRef<Type> result_types,
    ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) {
  // ONNXReluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                           ConstantOp 0,
  //                           %X)
  Value* operand = operands[0];
  auto zero = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  auto lessThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<SelectOp>(loc, lessThanZero, zero, operand);
  return result;
}

//===----------------------------------------------------------------------===//
// Element-wise n-ary ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseNaryOp, unsigned numArgs>
struct ONNXElementwiseNaryOpLowering : public ConversionPattern {
  ONNXElementwiseNaryOpLowering(MLIRContext* ctx)
      : ConversionPattern(ElementwiseNaryOp::getOperationName(), 1, ctx) {}
  PatternMatchResult matchAndRewrite(Operation* op, ArrayRef<Value*> operands,
      ConversionPatternRewriter& rewriter) const final {
    // TODO: Check that the types are valid.
    // An element-wise binary operation must have all operands and the result of
    // the same type. This should have been verified by the verifier.
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);

    // If the output has a dynamic dimension, pass the operands required for
    // each dynamic dimension to the AllocOp. The first operand of the binary
    // operation is used. The operands of the op need to match in terms of
    // dimensions with the result at this pre-optimization phase.
    // TODO: verify that dimensions match.
    // TODO: can the dimension of the result differ after optimizations?
    Value* alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, operands[0]);

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t rank = memRefShape.size();

    // Define loops.
    auto loopsOp = rewriter.create<KrnlDefineLoopsOp>(loc, rank);
    std::vector<Value*> originalLoops;
    originalLoops.reserve(rank);
    for (auto result : loopsOp.getResults()) {
      originalLoops.push_back(result);
    }

    // Define loop optimization.
    auto optimizedLoopsOp = rewriter.create<KrnlOptimizeLoopsOp>(loc, rank);
    std::vector<Value*> optimizedLoops;
    optimizedLoops.reserve(rank);
    for (auto result : optimizedLoopsOp.getResults()) {
      optimizedLoops.push_back(result);
    }
    Block& optimizationBlock = optimizedLoopsOp.region().front();

    KrnlIterateOperandPack pack(rewriter, originalLoops, optimizedLoops);
    // Iterate over the loop nest.
    // TODO (Tian): move this logic inside KrnlIterateOp. Pass MemRefShape
    // to KrnlIterateOp instead.
    for (int i = 0; i < rank; ++i) {
      if (memRefShape[i] < 0) {
        pack.pushConstantBound(0);
        pack.pushOperandBound(
            rewriter.create<DimOp>(loc, operands[0], i).getResult());
      } else {
        pack.pushConstantBound(0);
        pack.pushConstantBound(memRefShape[i]);
      }
    }

    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block& iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the
    // just generated instructions:

    // 1. Insert any optimizations in the KrnlOptimizeLoopsOp body.
    rewriter.setInsertionPointToEnd(&optimizationBlock);
    // Return from KrnlOptimizeLoopsOp body.
    // When no optimizations are present we just return the loops
    // unchaged.
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);
    rewriter.setInsertionPoint(optimizedLoopsOp);

    // 2. Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation:
    SmallVector<Value*, 4> loopIVs;
    for (auto arg : iterationBlock.getArguments())
      loopIVs.push_back(arg);

    SmallVector<Value*, numArgs> loadedVals;
    for (unsigned i = 0; i < numArgs; i++) {
      auto loadedVal = rewriter.create<LoadOp>(loc, operands[i], loopIVs);
      loadedVals.push_back(loadedVal);
    }

    auto loweredOpResult = mapToLowerScalarOp<ElementwiseNaryOp>(
        loc, memRefType.getElementType(), loadedVals, rewriter);

    // Store result in the resulting array.
    rewriter.create<StoreOp>(loc, loweredOpResult, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

template <typename ElementwiseNaryOp>
using ONNXElementwiseUnaryOpLowering =
    ONNXElementwiseNaryOpLowering<ElementwiseNaryOp, 1>;
template <typename ElementwiseNaryOp>
using ONNXElementwiseBinaryOpLowering =
    ONNXElementwiseNaryOpLowering<ElementwiseNaryOp, 2>;

//===----------------------------------------------------------------------===//
// Conversion from Tensor type to the Standard dialect MemRef type.
//===----------------------------------------------------------------------===//

struct TensorTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  LogicalResult convertType(Type t, SmallVectorImpl<Type>& results) override {
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
    return llvm::all_of(
        funcType.getInputs(), [this](Type type) { return isLegal(type); });
  }
};

}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
// Frontend to Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
namespace {
struct FrontendToKrnlLoweringPass
    : public ModulePass<FrontendToKrnlLoweringPass> {
  void runOnModule() final;
};
}  // end anonymous namespace.

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
  populateFuncOpTypeConversionPattern(
      patterns, &getContext(), tensor_to_memref_converter);

  // Frontent operation lowering.
  patterns.insert<ONNXElementwiseUnaryOpLowering<mlir::ONNXExpOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXTanhOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSinhOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCoshOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSigmoidOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXReluOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXAddOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXMulOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXDivOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXSubOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXAndOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXOrOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXXorOp>>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createLowerToKrnlPass() {
  return std::make_unique<FrontendToKrnlLoweringPass>();
}

static PassRegistration<FrontendToKrnlLoweringPass> pass(
    "lower-frontend", "Lower frontend ops to Krnl dialect.");
