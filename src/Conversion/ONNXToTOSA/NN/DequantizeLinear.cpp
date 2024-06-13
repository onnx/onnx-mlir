/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXDequantizeLinearOp.cpp - ONNXDequantizeLinearOp-----===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNXDequantizeLinearOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXDequantizeLinearOpLoweringToTOSA
    : public OpConversionPattern<ONNXDequantizeLinearOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXDequantizeLinearOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    Value x = op.getX();
    auto resultType = dyn_cast_if_present<ShapedType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          loc, "expected valid tensor result type");
    }

    if (auto zpTy = dyn_cast<ShapedType>(adaptor.getXZeroPoint().getType());
        !zpTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          loc, "expected zero point to have static shape");
    }

    if (auto zpTy = dyn_cast<ShapedType>(adaptor.getXScale().getType());
        !zpTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          loc, "expected scale to have static shape");
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
        rewriter, loc, adaptor.getXZeroPoint(), axis, resultType.getRank());
    auto scaleFactorConst = tosa::expandShape(
        rewriter, loc, adaptor.getXScale(), axis, resultType.getRank());

    // Dequantization formula is (x - zero_point) * scale
    // Cast into the destination type first

    // Cast the operands of (x - zero_point) to float32 to avoid underflows
    Type arithType = rewriter.getF32Type();
    Value subOpA = tosaBuilder.castToNewTensorElementType(x, arithType);
    Value subOpB = tosaBuilder.castToNewTensorElementType(zpConst, arithType);
    Value subOp = tosa::CreateOpAndInfer<mlir::tosa::SubOp>(
        rewriter, loc, subOpA.getType(), subOpA, subOpB)
                      .getResult();
    // There are no guarantees about the bitwith of the scale factor
    Value scaleFactorCast =
        tosaBuilder.castToNewTensorElementType(scaleFactorConst, arithType);
    Value mulOp = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(
        rewriter, loc, subOp.getType(), subOp, scaleFactorCast, 0)
                      .getResult();
    Value castOp = tosaBuilder.castToNewTensorElementType(
        mulOp, resultType.getElementType());

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXDequantizeLinearOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXDequantizeLinearOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
