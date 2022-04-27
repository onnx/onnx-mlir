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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

// This defines a template to construct ops whose legalizations are
// specialized.
template <typename OnnxOpT>
class ConvertOnnxOp : public OpConversionPattern<OnnxOpT> {
public:
  using OpConversionPattern<OnnxOpT>::OpConversionPattern;
  using OpAdaptor = typename OnnxOpT::Adaptor;
  LogicalResult matchAndRewrite(OnnxOpT op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

template <>
LogicalResult ConvertOnnxOp<ONNXReluOp>::matchAndRewrite(ONNXReluOp op,
    OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.X();
  auto inputTy = input.getType().dyn_cast<TensorType>();

  if (!inputTy)
    return op.emitError("Only Tensor types supported in TOSA");

  if (!inputTy.getElementType().isa<FloatType>()) {
    return op.emitError(
        "Only floating-point datatype legalization currently supported");
  }

  // Rescale the clampIn for quantized types. TBD
  // Maps to tosa.clamp which has both int and fp limits.
  Value clampIn = input;

  rewriter.replaceOpWithNewOp<tosa::ClampOp>(op, op.getType(), clampIn,
      rewriter.getI64IntegerAttr(0),
      rewriter.getI64IntegerAttr(std::numeric_limits<int32_t>::max()),
      rewriter.getF32FloatAttr(0.0f),
      rewriter.getF32FloatAttr(std::numeric_limits<float>::max()));
  return success();
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

  void runOnOperation() final;
};

void FrontendToTosaLoweringPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });

  target.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();

#define INSERT_ONNXOP_PATTERN(OnnxOp)                                          \
  target.addIllegalOp<OnnxOp>();                                               \
  patterns.add<ConvertOnnxOp<OnnxOp>>(typeConverter, context);
  INSERT_ONNXOP_PATTERN(ONNXReluOp);
#undef INSERT_ONNXOP_PATTERN

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createConvertONNXToTOSAPass() {
  return std::make_unique<FrontendToTosaLoweringPass>();
}

} // namespace onnx_mlir
