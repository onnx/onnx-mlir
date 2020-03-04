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

#include "src/conversion/onnx_to_krnl/onnx_to_krnl_common.hpp"

using namespace mlir;

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
// ConstantTensor operations are simply removed.
// The should be used to populate attributes and tensor dimensions of other
// operations (such as Reshape) and that information should have already been
// propagated in previous passes. This means that ConstantTensor operations
// are no longer used and can be removed.
//===----------------------------------------------------------------------===//

class ONNXConstantTensorLowering
    : public OpRewritePattern<ONNXConstantTensorOp> {
public:
  using OpRewritePattern<ONNXConstantTensorOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ONNXConstantTensorOp op,
                                     PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

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
  // Entry point
  patterns.insert<ONNXEntryPointLowering>(&getContext());
  patterns.insert<ONNXConstantTensorLowering>(&getContext());

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
