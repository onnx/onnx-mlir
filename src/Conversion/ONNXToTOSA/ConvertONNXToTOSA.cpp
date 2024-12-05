/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToTOSA.cpp - ONNX dialects to TOSA lowering -------===//
//
// Copyright (c) 2022 Arm Limited.
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace mlir;

namespace onnx_mlir {

void populateONNXToTOSAConversionPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx,
    int64_t groupedConvThreshold) {
  // Math
  populateLoweringONNXElementwiseOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXReduceOpsToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXGemmOpToTOSAPattern(target, patterns, typeConverter, ctx);
  populateLoweringONNXSoftmaxOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXConvOpToTOSAPattern(
      target, patterns, typeConverter, ctx, groupedConvThreshold);
  // Tensor
  populateLoweringONNXConcatOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXReshapeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXGatherOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXResizeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXShrinkOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXConstOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXEyeLikeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXPadOpToTOSAPattern(target, patterns, typeConverter, ctx);
  populateLoweringONNXFlattenOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXSliceOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXSplitOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXSqueezeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXTileOpToTOSAPattern(target, patterns, typeConverter, ctx);
  populateLoweringONNXExpandOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXTransposeOpToTOSAPattern(
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
  populateLoweringONNXMatMulOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXBatchNormalizationOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  // Flow
  populateLoweringONNXEntryPointOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXReshapeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXResizeOpToTOSAPattern(
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
    registry.insert<mlir::tosa::TosaDialect, mlir::shape::ShapeDialect>();
  }
  void runOnOperation() final;

public:
  Option<int64_t> groupedConvThreshold{*this, "grouped-conv-threshold",
      llvm::cl::desc("The threshold used to decompose grouped convolution "
                     "into a concatenation of tosa.conv2d operations"),
      llvm::cl::ZeroOrMore,
      llvm::cl::init(std::numeric_limits<int64_t>::max())};
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
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (isTOSAInt(type) || isa<FloatType>(type) || isa<NoneType>(type) ||
        isTOSABool(type))
      return type;
    return std::nullopt;
  });
  typeConverter.addConversion([&](TensorType type) -> std::optional<Type> {
    if (typeConverter.isLegal(type.getElementType()))
      return type;
    return std::nullopt;
  });

  // Define legal dialects and operations
  target.addLegalDialect<mlir::tosa::TosaDialect, func::FuncDialect,
      mlir::arith::ArithDialect, mlir::shape::ShapeDialect>();

  // Define patterns
  populateONNXToTOSAConversionPattern(
      target, patterns, typeConverter, context, groupedConvThreshold);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createConvertONNXToTOSAPass() {
  return std::make_unique<FrontendToTosaLoweringPass>();
}

} // namespace onnx_mlir
