//====------ ConvertONNXToKrnl.cpp - ONNX dialects to Krnl lowering -------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to a combination of
// Krnl IR and standard operations.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// EntryPoint Op lowering to Krnl Entry Point.
//===----------------------------------------------------------------------===//

class ONNXEntryPointLowering : public OpRewritePattern<ONNXEntryPointOp> {
public:
  using OpRewritePattern<ONNXEntryPointOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXEntryPointOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<KrnlEntryPointOp>(op,
        op.getAttrOfType<SymbolRefAttr>(
            ONNXEntryPointOp::getEntryPointFuncAttrName()),
        op.getAttrOfType<IntegerAttr>(ONNXEntryPointOp::getNumInputsAttrName()),
        op.getAttrOfType<IntegerAttr>(
            ONNXEntryPointOp::getNumOutputsAttrName()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FuncOp lowering to Function with init and main blocks.
//===----------------------------------------------------------------------===//

struct FuncOpSignatureConversion : public OpConversionPattern<FuncOp> {
  FuncOpSignatureConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(converter, ctx) {}

  /// Hook for derived classes to implement combined matching and rewriting.
  LogicalResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType type = funcOp.getType();

    // Convert the original function types.
    TypeConverter::SignatureConversion result(type.getNumInputs());
    SmallVector<Type, 1> newResults;
    if (failed(typeConverter->convertSignatureArgs(type.getInputs(), result)) ||
        failed(typeConverter->convertTypes(type.getResults(), newResults)) ||
        failed(rewriter.convertRegionTypes(&funcOp.getBody(), *typeConverter,
                                           &result)))
      return failure();

    addInitBlock(rewriter, funcOp.getLoc(), funcOp);

    // Update the function signature in-place.
    rewriter.updateRootInPlace(funcOp, [&] {
      funcOp.setType(FunctionType::get(result.getConvertedTypes(), newResults,
                                       funcOp.getContext()));
    });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Frontend to Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
namespace {
struct FrontendToKrnlLoweringPass
    : public PassWrapper<FrontendToKrnlLoweringPass, OperationPass<ModuleOp>> {
  void runOnOperation() final;
};
} // end anonymous namespace.

void FrontendToKrnlLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // // Create a new initMap
  // initMap = new llvm::DenseMap<FuncOp, ONNXOperandsInitState*>();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<KrnlOpsDialect, AffineDialect, StandardOpsDialect>();

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

  // // Type conversion for function signatures.
  // // Call MLIR FuncOp signature conversion when result type is
  // // a ranked tensor.
  // populateFuncOpTypeConversionPattern(
  //     patterns, &getContext(), tensor_to_memref_converter);

  // Frontend operation lowering.
  // Math
  populateLoweringONNXElementwiseOpPattern(patterns, &getContext());
  populateLoweringONNXGemmOpPattern(patterns, &getContext());
  populateLoweringONNXReductionOpPattern(patterns, &getContext());
  populateLoweringONNXSoftmaxOpPattern(patterns, &getContext());
  populateLoweringONNXMatMulOpPattern(patterns, &getContext());
  // Tensor
  populateLoweringONNXReshapeOpPattern(patterns, &getContext());
  populateLoweringONNXPadConstantValuePadOpPattern(patterns, &getContext());
  populateLoweringONNXPadOpPattern(patterns, &getContext());
  populateLoweringONNXUnsqueezeOpPattern(patterns, &getContext());
  populateLoweringONNXTransposeOpPattern(patterns, &getContext());
  populateLoweringONNXIdentityOpPattern(patterns, &getContext());
  populateLoweringONNXConstantOpPattern(patterns, &getContext());
  populateLoweringONNXConcatOpPattern(patterns, &getContext());
  populateLoweringONNXSqueezeOpPattern(patterns, &getContext());
  populateLoweringONNXSplitOpPattern(patterns, &getContext());
  // Neural network
  populateLoweringONNXConvOpPattern(patterns, &getContext());
  populateLoweringONNXNormalizationOpPattern(patterns, &getContext());
  populateLoweringONNXPoolingOpPattern(patterns, &getContext());
  // Recurrent neural network
  populateLoweringONNXLSTMOpPattern(patterns, &getContext());
  // Entry point
  patterns.insert<ONNXEntryPointLowering>(&getContext());
  patterns.insert<FuncOpSignatureConversion>(
      &getContext(), tensor_to_memref_converter);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, patterns)))
    signalPassFailure();

  initMap.clear();
}

std::unique_ptr<Pass> mlir::createLowerToKrnlPass() {
  return std::make_unique<FrontendToKrnlLoweringPass>();
}
