// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "mlir/Support/LogicalResult.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXEyeLikeLoweringToTOSA : public OpConversionPattern<ONNXEyeLikeOp> {
public:
  using OpConversionPattern<ONNXEyeLikeOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXEyeLikeOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXEyeLikeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "onnx.EyeLikeOp needs to have static shape for lowering to tosa");
    }
    const auto elementType = resultType.getElementType();
    const auto convertedType = dyn_cast_or_null<ShapedType>(
        getTypeConverter()->convertType(resultType));
    if (!convertedType) {
      return rewriter.notifyMatchFailure(
          op, "EyeLike type not supported in tosa");
    }
    int64_t k = 0;
    if (auto kAttr = adaptor.getKAttr()) {
      k = kAttr.getSInt();
    }
    DenseElementsAttr replacementAttr;
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      replacementAttr = getEyeLikeAttr(convertedType, resultType.getDimSize(0),
          resultType.getDimSize(1), k, APInt(intType.getWidth(), 0),
          APInt(intType.getWidth(), 1));
    } else if (auto floatType = dyn_cast<FloatType>(elementType)) {
      replacementAttr = getEyeLikeAttr(convertedType, resultType.getDimSize(0),
          resultType.getDimSize(1), k,
          APFloat::getZero(floatType.getFloatSemantics()),
          APFloat(floatType.getFloatSemantics(), 1));
    } else {
      return rewriter.notifyMatchFailure(op, "Only int and float supported");
    }

    rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(
        op, convertedType, replacementAttr);
    return success();
  }

private:
  template <typename T>
  DenseElementsAttr getEyeLikeAttr(const ShapedType type, const int64_t dimY,
      const int64_t dimX, const int64_t k, const T zero, const T one) const {
    const auto size = dimX * dimY;
    SmallVector<T, 0> vec(size, zero);
    const auto smallDim = std::min(dimX - std::abs(k), dimY);
    for (int64_t i = 0; i < smallDim; ++i) {
      if (k >= 0) {
        vec[(dimX * i) + i + k] = one;
      } else {
        vec[(dimX * (i - k)) + i] = one;
      }
    }
    return DenseElementsAttr::get(type, vec);
  }
};
} // namespace

void populateLoweringONNXEyeLikeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXEyeLikeLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
