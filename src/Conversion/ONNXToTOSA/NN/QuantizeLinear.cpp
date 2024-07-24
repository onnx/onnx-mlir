/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXQuantizeLinearOp.cpp - ONNXQuantizeLinearOp---------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNXQuantizeLinearOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXQuantizeLinearOpLoweringToTOSA
    : public OpConversionPattern<ONNXQuantizeLinearOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXQuantizeLinearOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultType = dyn_cast_if_present<ShapedType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          loc, "expected valid tensor result type");
    }

    if (auto zpTy = dyn_cast<ShapedType>(adaptor.getYZeroPoint().getType());
        zpTy && !zpTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          loc, "expected zero point to have static shape");
    }

    if (auto zpTy = dyn_cast<ShapedType>(adaptor.getYScale().getType());
        zpTy && !zpTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          loc, "expected scale to have static shape");
    }

    if (!op.getSaturate()) {
      return rewriter.notifyMatchFailure(loc, "Only saturate=1 is supported");
    }

    int64_t axis = op.getAxis();
    // See https://github.com/onnx/onnx/issues/6067
    if (axis == 1 && resultType.getRank() == 1)
      axis = 0;
    if (axis < -resultType.getRank() || axis >= resultType.getRank()) {
      return rewriter.notifyMatchFailure(loc, "axis is invalid");
    }
    if (axis < 0)
      axis += resultType.getRank();

    Value x = adaptor.getX();
    Type xType = x.getType();

    // Quantization formula is saturate((x / y_scale) + y_zero_point)
    // tosa.mul doesn't allow different ranks
    auto expandedScaleFactorConst = tosa::expandShape(
        rewriter, loc, adaptor.getYScale(), axis, resultType.getRank());
    // Replace the division by a reciprocal followed by a mul
    Value recOp = tosa::CreateOpAndInfer<mlir::tosa::ReciprocalOp>(rewriter,
        loc, expandedScaleFactorConst.getType(), expandedScaleFactorConst)
                      .getResult();
    Value scaledResult = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(
        rewriter, loc, xType, x, recOp, 0)
                             .getResult();

    // Quantization to i4/i8/16/ is particular since the intermediate result of
    // (x / y_scale) must round to the nearest even. This is particularly
    // important if the intermediate result is e.g. 8.5. If we don't round to
    // the nearest even before adding the (potentially odd) zero point, we would
    // end up with a different result
    bool quantizingToInt = isa<IntegerType>(resultType.getElementType());
    if (quantizingToInt) {
      // ONNX QuantizeLinear op supports those integer zero point types:
      // int16, int4, int8, uint16, uint4, uint8
      // Convert the scaled result to a safe bitwith (i32) that avoids
      // underflows/overflows
      scaledResult = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(rewriter, loc,
          resultType.cloneWith({}, rewriter.getI32Type()), scaledResult)
                         .getResult();
    }

    // If there is no zero point, we are done
    if (isa<NoneType>(adaptor.getYZeroPoint().getType())) {
      Value result = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
          rewriter, loc, resultType, scaledResult)
                         .getResult();
      rewriter.replaceOp(op, result);
      return success();
    }

    Value expandedZpConst = tosa::expandShape(
        rewriter, loc, adaptor.getYZeroPoint(), axis, resultType.getRank());

    // Cast the expandedZpConst to have the same rank and element type as
    // the scaledResult. tosa.add doesn't allow different ranks
    Value castedZp;
    if (quantizingToInt) {
      castedZp = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(rewriter, loc,
          cast<ShapedType>(expandedZpConst.getType())
              .cloneWith({}, rewriter.getI32Type()),
          expandedZpConst)
                     .getResult();
    } else {
      // zpConst has the same type as the result of QLinear which is always
      // smaller than the input type. Cast it to the input type.
      castedZp = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
          rewriter, loc, expandedScaleFactorConst.getType(), expandedZpConst)
                     .getResult();
    }

    Value addOp = tosa::CreateOpAndInfer<mlir::tosa::AddOp>(
        rewriter, loc, scaledResult.getType(), scaledResult, castedZp)
                      .getResult();

    Value clampedRes = addOp;
    if (quantizingToInt) {
      // If the destination type is an integer, perform saturation.
      IntegerType resTypeInt =
          dyn_cast<IntegerType>(resultType.getElementType());

      // Compute the max/min values for the said type from the 64-bit max
      auto width = resTypeInt.getIntOrFloatBitWidth();
      APInt maxVal = resTypeInt.isUnsigned() ? APInt::getMaxValue(width)
                                             : APInt::getSignedMaxValue(width);
      APInt minVal = resTypeInt.isUnsigned() ? APInt::getZero(width)
                                             : APInt::getSignedMinValue(width);

      clampedRes = tosa::CreateOpAndInfer<mlir::tosa::ClampOp>(rewriter, loc,
          addOp.getType(), addOp,
          rewriter.getIntegerAttr(rewriter.getI64Type(), minVal.sext(64)),
          rewriter.getIntegerAttr(rewriter.getI64Type(), maxVal.zext(64)),
          // We ignore floating point values, we're clamping integers.
          rewriter.getFloatAttr(
              rewriter.getF32Type(), (float)(minVal.getSExtValue())),
          rewriter.getFloatAttr(
              rewriter.getF32Type(), (float)(maxVal.getZExtValue())));
    }

    // Cast into the result type
    Value result = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, loc, resultType, clampedRes)
                       .getResult();

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXQuantizeLinearOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXQuantizeLinearOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
