/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToTOSA.cpp - ONNX dialects to TOSA lowering -------===//
//
// Copyright (c) 2022 Arm Limited.
// Copyright (c) 2022-2026 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
using namespace mlir;

namespace onnx_mlir {

void populateONNXToTOSAConversionPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx,
    int64_t groupedConvThreshold, bool convertSliceOnlyWhenStepOne) {
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
      target, patterns, typeConverter, ctx, convertSliceOnlyWhenStepOne);
  populateLoweringONNXSplitOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXSqueezeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXTileOpToTOSAPattern(target, patterns, typeConverter, ctx);
  populateLoweringONNXExpandOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXTransposeOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  populateLoweringONNXWhereOpToTOSAPattern(
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
  Option<bool> convertSliceOnlyWhenStepOne{*this,
      "convert-slice-only-when-step-one",
      llvm::cl::desc("If enabled, convert onnx.slice only if all steps are 1"),
      llvm::cl::ZeroOrMore, llvm::cl::init(false)};
  ListOption<std::string> excludedOps{*this, "excluded-ops",
      llvm::cl::desc("ONNX op names to exclude from TOSA conversion "
                     "(e.g. Gather,Cast)"),
      llvm::cl::ZeroOrMore};
};

Value handleDynamicToStaticShapes(OpBuilder & /*builder*/, TensorType type,
    ValueRange inputs, Location /*loc*/) {
  if (inputs.size() != 1) {
    // we can only handle inputs of size 1
    return {};
  }
  Value out = inputs.front();
  auto outType = dyn_cast<ShapedType>(out.getType());
  if (!outType) {
    // no operation as input
    return {};
  }
  if (outType.hasStaticShape() || !type.hasStaticShape()) {
    // we are looking for a dynamic to static "cast"
    return {};
  }
  if (outType.getElementType() != type.getElementType()) {
    // type is not equal
    // this should be mostly caught by the operation validation already
    return {};
  }
  if (outType.hasRank()) {
    if (outType.getRank() != type.getRank()) {
      // rank does not match
      // this seems to be caught by the shape inference already
      return {};
    } else {
      auto outShape = outType.getShape();
      auto tyShape = type.getShape();

      for (int i = 0; i < outType.getRank(); ++i) {
        if (outType.isDynamicDim(i)) {
          continue;
        }
        if (outShape[i] != tyShape[i]) {
          // input and output does not match in one dimension
          return {};
        }
      }
    }
  }
  out.setType(type);
  return out;
}

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
  typeConverter.addSourceMaterialization(handleDynamicToStaticShapes);

  // Define legal dialects and operations
  target.addLegalDialect<mlir::tosa::TosaDialect, func::FuncDialect,
      mlir::arith::ArithDialect, mlir::shape::ShapeDialect,
      mlir::affine::AffineDialect>();

  for (const std::string &opName : excludedOps)
    target.addLegalOp(OperationName("onnx." + opName, context));

  // Define patterns
  populateONNXToTOSAConversionPattern(target, patterns, typeConverter, context,
      groupedConvThreshold, convertSliceOnlyWhenStepOne);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createConvertONNXToTOSAPass() {
  return std::make_unique<FrontendToTosaLoweringPass>();
}

} // namespace onnx_mlir
