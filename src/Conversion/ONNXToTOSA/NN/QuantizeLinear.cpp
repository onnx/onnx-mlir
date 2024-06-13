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
        !zpTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          loc, "expected zero point to have static shape");
    }

    if (auto zpTy = dyn_cast<ShapedType>(adaptor.getYScale().getType());
        !zpTy.hasStaticShape()) {
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

    // Since tosa.add and tosa.mul don't allow different ranks, get the value
    // from the constants, and create a new constant of the same rank as the
    // input out of it in order to have a correct add and mul.
    auto zpConst = tosa::expandShape(
        rewriter, loc, adaptor.getYZeroPoint(), axis, resultType.getRank());
    auto scaleFactorConst = tosa::expandShape(
        rewriter, loc, adaptor.getYScale(), axis, resultType.getRank());

    Value x = adaptor.getX();
    Type xType = x.getType();

    // Quantization formula is ((x / y_scale) + y_zero_point)
    // Replace the division by a reciprocal followed by a mul
    Value recOp = tosa::CreateOpAndInfer<mlir::tosa::ReciprocalOp>(
        rewriter, loc, scaleFactorConst.getType(), scaleFactorConst)
                      .getResult();
    Value mulOp = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(
        rewriter, loc, xType, x, recOp, 0)
                      .getResult();
    // zpConst has the same type as the result of QLinear which is always
    // smaller than the input type. Cast it to the input type.
    Value castZp = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, loc, scaleFactorConst.getType(), zpConst)
                       .getResult();
    Value addOp = tosa::CreateOpAndInfer<mlir::tosa::AddOp>(
        rewriter, loc, xType, mulOp, castZp)
                      .getResult();
    // Cast into the result type
    Value castOp = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, loc, resultType, addOp)
                       .getResult();

    rewriter.replaceOp(op, castOp);
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
