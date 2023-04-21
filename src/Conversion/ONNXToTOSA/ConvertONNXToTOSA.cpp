/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToTOSA.cpp - ONNX dialects to TOSA lowering -------===//
//
// Copyright (c) 2022 Arm Limited.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace mlir;

namespace onnx_mlir {

void populateONNXToTOSAConversionPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  // Math
  populateLoweringONNXElementwiseOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXGemmOpToTOSAPattern(target, patterns, typeConverter, ctx);
  populateLoweringONNXSoftmaxOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXConvOpToTOSAPattern(target, patterns, typeConverter, ctx);
  populateLoweringONNXReduceMeanOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  // Tensor
  populateLoweringONNXConcatOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXReshapeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXGatherOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXResizeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXConstOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXPadOpToTOSAPattern(target, patterns, typeConverter, ctx);
  populateLoweringONNXFlattenOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXPadOpToTOSAPattern(target, patterns, typeConverter, ctx);
  populateLoweringONNXSliceOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXTransposeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXUnsqueezeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  // NN
  populateLoweringONNXMaxPoolSingleOutOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXAveragePoolOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXQuantizeLinearOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXDequantizeLinearOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  // Flow
  populateLoweringONNXEntryPointOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
}

// Performs lowering to TOSA dialect
struct FrontendToTosaLoweringPass
    : public PassWrapper<FrontendToTosaLoweringPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "convert-onnx-to-tosa"; }

  StringRef getDescription() const override {
    return "Lower frontend ops to TOSA dialect.";
  }

  FrontendToTosaLoweringPass() = default;
  FrontendToTosaLoweringPass(const FrontendToTosaLoweringPass &pass)
      : PassWrapper<FrontendToTosaLoweringPass, OperationPass<ModuleOp>>() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::tosa::TosaDialect>();
  }
  void runOnOperation() final;
};

void FrontendToTosaLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();
  // Define final conversion target
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);

  // We use the type converter to legalize types before any conversion patterns
  // are executed. This ensures that we do not need to trigger separate
  // conversion failures. Quantized types are not supported right now.
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> Optional<Type> {
    if (isTOSASignedInt(type) || isTOSAFloat(type) ||
        type.isa<mlir::NoneType>())
      return type;
    return llvm::None;
  });
  typeConverter.addConversion([&](TensorType type) -> Optional<Type> {
    if (typeConverter.isLegal(type.getElementType()))
      return type;
    return llvm::None;
  });

  // Define legal dialects and operations
  target.addLegalDialect<mlir::tosa::TosaDialect, mlir::func::FuncDialect,
      mlir::arith::ArithDialect>();

  // Define patterns
  populateONNXToTOSAConversionPattern(target, patterns, typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createConvertONNXToTOSAPass() {
  return std::make_unique<FrontendToTosaLoweringPass>();
}

} // namespace onnx_mlir
