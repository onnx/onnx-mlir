/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====- ConvertONNXToStablehlo.cpp - ONNX dialects to Stablehlo lowering --===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file implements the lowering of frontend operations to a combination of
// Stablehlo IR and standard operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

void populateONNXToStablehloConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, bool enableUnroll) {
  // Math
  populateLoweringONNXClipOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXElementwiseOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXGemmOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXMatMulOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXReductionOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXSoftmaxOpToStablehloPattern(patterns, ctx);
  // Neural network
  populateLoweringONNXConvOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXConvTransposeOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXNormalizationOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXPoolingOpToStablehloPattern(patterns, ctx);
  // Recurrent neural network
  populateLoweringONNXLSTMOpToStablehloPattern(patterns, ctx, enableUnroll);
  // Tensor
  populateLoweringONNXArgMaxOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXConcatOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXConstantOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXDepthToSpaceOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXDimOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXExpandOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXFlattenOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXGatherOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXGatherElementsOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXIdentityOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXOneHotOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXPadOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXReshapeOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXScatterNDOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXShapeOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXSliceOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXSplitOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXSqueezeOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXTileOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXTransposeOpToStablehloPattern(patterns, ctx);
  populateLoweringONNXUnsqueezeOpToStablehloPattern(patterns, ctx);
}

//===----------------------------------------------------------------------===//
// Frontend to Stablehlo Dialect lowering pass
//===----------------------------------------------------------------------===//

struct FrontendToStablehloLoweringPass
    : public PassWrapper<FrontendToStablehloLoweringPass,
          OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-onnx-to-stablehlo"; }

  StringRef getDescription() const override {
    return "Lower frontend ops to Stablehlo dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  FrontendToStablehloLoweringPass() = default;
  FrontendToStablehloLoweringPass(const FrontendToStablehloLoweringPass &pass)
      : PassWrapper<FrontendToStablehloLoweringPass,
            OperationPass<ModuleOp>>() {}
  FrontendToStablehloLoweringPass(bool enableUnroll) {
    // Below, need explicit assignment to enable implicit conversion of bool
    // to Option<bool>.
    this->enableUnroll = enableUnroll;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<shape::ShapeDialect>();
  }

  void runOnOperation() final;

public:
  // Some ops (RNN ops for example) will have loops inside them. We can
  // choose to unroll the loop, which means to expand the loops completely
  // so there are no loops left, or to rewrite the loop into stablehlo::WhileOp
  Option<bool> enableUnroll{*this, "enable-unroll",
      llvm::cl::desc(
          "Enable unroll rather than lowering to stablehlo::WhileOp."),
      llvm::cl::init(true)};
};

void FrontendToStablehloLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  // Added affine as some affine maps are generated by IndexExpression. It could
  // be disabled and/or replaced by shape max/min.
  target.addLegalDialect<stablehlo::StablehloDialect, func::FuncDialect,
      arith::ArithDialect, shape::ShapeDialect, mlir::affine::AffineDialect,
      tensor::TensorDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

  // Define patterns.
  populateONNXToStablehloConversionPattern(
      patterns, &getContext(), enableUnroll);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLowerToStablehloPass() {
  return std::make_unique<FrontendToStablehloLoweringPass>();
}

std::unique_ptr<Pass> createLowerToStablehloPass(bool enableUnroll) {
  return std::make_unique<FrontendToStablehloLoweringPass>(enableUnroll);
}

} // namespace onnx_mlir
