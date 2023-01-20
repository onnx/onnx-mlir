/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToTorch.cpp - ONNX dialects to Torch lowering
//-------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// ===============================================================================
//
// This file implements the lowering of frontend operations to Torch backend IR.
//
//===------------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

void populateONNXToTorchConversionPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  // Math
  populateLoweringONNXElementwiseOpToTorchPattern(typeConverter, patterns, ctx);
  populateLoweringONNXConstantOpToTorchPattern(typeConverter, patterns, ctx);
}

//===----------------------------------------------------------------------===//
// Frontend to Mhlo Dialect lowering pass
//===----------------------------------------------------------------------===//

struct FrontendToTorchLoweringPass
    : public PassWrapper<FrontendToTorchLoweringPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-onnx-to-torch"; }

  StringRef getDescription() const override {
    return "Lower frontend ops to Torch dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  FrontendToTorchLoweringPass() = default;
  FrontendToTorchLoweringPass(const FrontendToTorchLoweringPass &pass)
      : PassWrapper<FrontendToTorchLoweringPass, OperationPass<ModuleOp>>() {}

  void runOnOperation() final;
};

void FrontendToTorchLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<torch::Torch::TorchDialect, func::FuncDialect>();

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  onnx_mlir::setupTorchTypeConversion(target, typeConverter);

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

  // Define patterns.
  populateONNXToTorchConversionPattern(typeConverter, patterns, &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLowerToTorchPass() {
  return std::make_unique<FrontendToTorchLoweringPass>();
}

} // namespace onnx_mlir
