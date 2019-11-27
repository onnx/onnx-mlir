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
static Value* insertAllocAndDealloc(
    MemRefType type, Location loc, PatternRewriter& rewriter,
    Value *oldMemRef = nullptr) {
  // Put together alloc operands for any dynamic dimensions of the memref.
  AllocOp alloc;
  if (oldMemRef) {
    SmallVector<Value *, 4> allocOperands;
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

  return alloc;
}

namespace {

//===----------------------------------------------------------------------===//
// AddOp lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
struct ONNXAddOpLowering : public ConversionPattern {
  ONNXAddOpLowering(MLIRContext* ctx)
      : ConversionPattern(mlir::ONNXAddOp::getOperationName(), 1, ctx) {}

  PatternMatchResult matchAndRewrite(Operation* op, ArrayRef<Value*> operands,
      ConversionPatternRewriter& rewriter) const final {
    // TODO: Check that the types are valid.
    // Add is an operation that must have all operands and the result of
    // the same type. This should have been verified by the verifier.
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);

    // If the output has a dynamic dimension, pass the operands required for
    // each dynamic dimension to the AllocOp. The first operand of the Add
    // operation is used. The operands of the Add need to match in terms of
    // dimensions with the result at this pre-optimization phase.
    // TODO: verify that dimensions match.
    // TODO: can the dimension of the result differ after optimizations?
    Value *alloc;
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
    else
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, operands[0]);

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

    // Iterate over the loop nest.
    // TODO (Tian): move this logic inside KrnlIterateOp. Pass MemRefShape
    // to KrnlIterateOp instead.
    SmallVector<Value*, 8> operandBounds;
    SmallVector<int64_t, 8> constBounds;
    SmallVector<int, 16> boundTypes;
    for (int i = 0; i < rank; ++i) {
      if (memRefShape[i] < 0) {
        // This is a dynamic value, hence use operands.
        // Lower bound
        constBounds.push_back(0);
        boundTypes.push_back(0);
        // Upper bound
        operandBounds.push_back(
            rewriter.create<DimOp>(loc, operands[0], i).getResult());
        boundTypes.push_back(1);
      } else {
        // Lower bound
        constBounds.push_back(0);
        boundTypes.push_back(0);
        // Upper bound
        constBounds.push_back(memRefShape[i]);
        boundTypes.push_back(0);
      }
    }
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, originalLoops,
        optimizedLoops, operandBounds, constBounds, boundTypes);
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

    // Handle AddOp:
    SmallVector<Value*, 4> loopIVs;
    for (auto arg : iterationBlock.getArguments())
      loopIVs.push_back(arg);
    auto loadedFirstVal =
        rewriter.create<LoadOp>(loc, operands[0], loopIVs);
    auto loadedSecondVal =
        rewriter.create<LoadOp>(loc, operands[1], loopIVs);

    // TODO: Choose type of the Add for now use the Float Add.
    auto addOpResult = rewriter.create<AddFOp>(
        loc, loadedFirstVal, loadedSecondVal);

    // Store result in the resulting array.
    rewriter.create<StoreOp>(loc, addOpResult, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

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
    return llvm::all_of(funcType.getInputs(),
        [this](Type type) { return isLegal(type); });
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
  patterns.insert<ONNXAddOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(
          module, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createLowerToKrnlPass() {
  return std::make_unique<FrontendToKrnlLoweringPass>();
}

static PassRegistration<FrontendToKrnlLoweringPass> pass(
     "lower-frontend", "Lower frontend ops to Krnl dialect.");
