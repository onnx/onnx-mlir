/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXQuantizeLinearOp.cpp - ONNXQuantizeLinearOp---------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNXBatchNormalizationOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;
namespace onnx_mlir {
namespace {

Value legalizeBatchNorm(Operation *op, ConversionPatternRewriter &rewriter,
    Value input, Value mean, Value scale, Value bias, Value var, APFloat eps) {

  TosaBuilder tosaBuilder(rewriter, op->getLoc());
  // reshape rank-1 tensors (scale, bias, mean, variance),
  // such that they have the same rank as input/output tensor
  Value reshapedMean;
  Value reshapedScale;
  Value reshapedBias;
  Value reshapedVar;

  auto outType = op->getResult(0).getType();
  auto outTensorType = cast<RankedTensorType>(outType);

  // The layout of the output is N x C x D1 x D2 … Dn. For batch
  // normalization, the C dimension is kept. The new shape should be {1, C, 1,
  // 1, ...}.
  SmallVector<int64_t> newShape = {1, outTensorType.getShape()[1]};
  for (auto i = 2; i < outTensorType.getRank(); i++)
    newShape.push_back(1);

  reshapedMean = tosaBuilder.reshape(mean, ArrayRef<int64_t>(newShape));
  reshapedScale = tosaBuilder.reshape(scale, ArrayRef<int64_t>(newShape));
  reshapedBias = tosaBuilder.reshape(bias, ArrayRef<int64_t>(newShape));
  reshapedVar = tosaBuilder.reshape(var, ArrayRef<int64_t>(newShape));

  // epsilon's shape: constant -> {1, 1, 1, ...}
  newShape[1] = 1;
  auto epsVal = tosaBuilder.getSplattedConst(
      eps.convertToFloat(), newShape, outTensorType.getElementType());

  // output = (input - mean) * scale * rsqrt(var + eps) + bias
  auto op1SubInputMean =
      tosaBuilder.binaryOp<mlir::tosa::SubOp>(input, reshapedMean);
  auto op2AddVarEps =
      tosaBuilder.binaryOp<mlir::tosa::AddOp>(reshapedVar, epsVal);
  auto op3RsqrtOp2 = tosaBuilder.unaryOp<mlir::tosa::RsqrtOp>(op2AddVarEps);
  auto op4MulOp1Op3 = tosaBuilder.mul(op1SubInputMean, op3RsqrtOp2, 0);
  auto op5MulOp4Scale = tosaBuilder.mul(op4MulOp1Op3, reshapedScale, 0);
  return tosaBuilder.binaryOp<mlir::tosa::AddOp>(op5MulOp4Scale, reshapedBias);
}

class ONNXBatchNormalizationOpLoweringToTOSA
    : public OpConversionPattern<ONNXBatchNormalizationOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXBatchNormalizationOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXBatchNormalizationOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto meanRes = op.getResult(1);
    auto varRes = op.getResult(2);
    if (!meanRes.use_empty() || !varRes.use_empty()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "mean and variance must have no use.");
    }

    Value input = op.getX();
    Value mean = op.getInputMean();
    Value scale = op.getScale();
    Value bias = op.getB();
    Value var = op.getInputVar();

    auto meanTy = cast<ShapedType>(meanRes.getType());
    auto varTy = cast<ShapedType>(varRes.getType());

    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    // Since Mean and Variance results have no use, represent them as
    // zero-splatted constants. They will be optimized out later.
    rewriter.replaceOp(op, {legalizeBatchNorm(op, rewriter, input, mean, scale,
                                bias, var, op.getEpsilon()),
                               tosaBuilder.getSplattedConst(0.0,
                                   meanTy.getShape(), meanTy.getElementType()),
                               tosaBuilder.getSplattedConst(0.0,
                                   varTy.getShape(), varTy.getElementType())});
    return success();
  }
};

class ONNXBatchNormalizationInferenceModeOpLoweringToTOSA
    : public OpConversionPattern<ONNXBatchNormalizationInferenceModeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXBatchNormalizationInferenceModeOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXBatchNormalizationInferenceModeOp op,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

    Value input = op.getX();
    Value mean = op.getMean();
    Value scale = op.getScale();
    Value bias = op.getB();
    Value var = op.getVar();

    rewriter.replaceOp(op, {legalizeBatchNorm(op, rewriter, input, mean, scale,
                               bias, var, op.getEpsilon())});
    return success();
  }
};
} // namespace

void populateLoweringONNXBatchNormalizationOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXBatchNormalizationInferenceModeOpLoweringToTOSA,
      ONNXBatchNormalizationOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
