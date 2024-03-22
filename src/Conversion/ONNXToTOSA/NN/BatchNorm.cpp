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
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// output = (input - mean) * scale / sqrt(var + eps) + bias
Value computeBatchNorm(Operation *op, ConversionPatternRewriter &rewriter,
    Type outType, Value input, Value variance, Value eps, Value mean,
    Value scale, Value bias) {
  auto op1SubInputMean =
      rewriter.create<mlir::tosa::SubOp>(op->getLoc(), outType, input, mean);
  auto op2AddVarEps = rewriter.create<mlir::tosa::AddOp>(
      op->getLoc(), variance.getType(), variance, eps);
  auto op3RsqrtOp2 = rewriter.create<mlir::tosa::RsqrtOp>(
      op->getLoc(), variance.getType(), op2AddVarEps.getResult());
  auto op4MulOp1Op3 = rewriter.create<mlir::tosa::MulOp>(op->getLoc(), outType,
      op1SubInputMean.getResult(), op3RsqrtOp2.getResult(), 0);
  auto op5MulOp4Scale = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), outType, op4MulOp1Op3.getResult(), scale, 0);
  return rewriter
      .create<mlir::tosa::AddOp>(
          op->getLoc(), outType, op5MulOp4Scale.getResult(), bias)
      .getResult();
}

class ONNXBatchNormalizationInferenceModeOpLoweringToTOSA
    : public OpConversionPattern<ONNXBatchNormalizationInferenceModeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXBatchNormalizationInferenceModeOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXBatchNormalizationInferenceModeOp op,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

    auto outType = getTypeConverter()->convertType(op.getResult().getType());

    // reshape rank-1 tensors (scale, bias, mean, variance) and attribute
    // epsilon, such that they have the same rank as input/output tensor
    auto reshapeToInputDim = [&](Operation *op,
                                 ConversionPatternRewriter &rewriter,
                                 Type outType, const Value toBcast,
                                 Value &result) {
      auto toBCastType = toBcast.getType().dyn_cast<RankedTensorType>();
      if (toBCastType.getRank() > 1) {
        return rewriter.notifyMatchFailure(op, "Rank cannot be more than 1");
      }
      auto outTensorType = outType.cast<RankedTensorType>();

      // onnx-mlir uses layout NCHW for input. For batch normalization, the C
      // dimension is kept. The new shape should be {1, C, 1, 1}, if the rank of
      // input/output tensor is 4
      SmallVector<int64_t> newShape = {1, toBCastType.getShape()[0]};
      for (auto i = 2; i < outTensorType.getRank(); i++) {
        newShape.push_back(1);
      }

      auto newType =
          RankedTensorType::get(newShape, outTensorType.getElementType());
      result =
          tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(rewriter, op->getLoc(),
              newType, toBcast, rewriter.getDenseI64ArrayAttr(newShape));
      if (!result) {
        return failure();
      }
      return success();
    };

    Value reshapedInputMean;
    Value reshapedScale;
    Value reshapedB;
    Value reshapedInputVar;

    if (failed(reshapeToInputDim(op.getOperation(), rewriter, outType,
            op.getMean(), reshapedInputMean))) {
      return rewriter.notifyMatchFailure(op, "failed to reshape mean");
    }

    if (failed(reshapeToInputDim(op.getOperation(), rewriter, outType,
            op.getScale(), reshapedScale))) {
      return rewriter.notifyMatchFailure(op, "failed to reshape scale");
    }

    if (failed(reshapeToInputDim(
            op.getOperation(), rewriter, outType, op.getB(), reshapedB))) {
      return rewriter.notifyMatchFailure(op, "failed to reshape bias");
    }

    if (failed(reshapeToInputDim(op.getOperation(), rewriter, outType,
            op.getVar(), reshapedInputVar))) {
      return rewriter.notifyMatchFailure(op, "failed to reshape variance");
    }

    Value reshapedEps;
    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    auto Eps = tosaBuilder.getConst(
        llvm::SmallVector<float>{op.getEpsilon().convertToFloat()}, {1});
    if (failed(reshapeToInputDim(
            op.getOperation(), rewriter, outType, Eps, reshapedEps))) {
      return rewriter.notifyMatchFailure(op, "failed to reshape epsilon");
    }

    auto batchNorm =
        computeBatchNorm(op, rewriter, outType, op.getX(), reshapedInputVar,
            reshapedEps, reshapedInputMean, reshapedScale, reshapedB);

    rewriter.replaceOp(op, batchNorm);
    return success();
  }
};
} // namespace

void populateLoweringONNXBatchNormalizationOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXBatchNormalizationInferenceModeOpLoweringToTOSA>(
      typeConverter, ctx);
}

} // namespace onnx_mlir