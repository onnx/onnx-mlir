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

    auto zeroPoint = adaptor.getXZeroPoint();
    auto zpTy = zeroPoint.getType();
    if (isa<NoneType>(zpTy)) {
      zeroPoint = {};
    } else if (auto shapedTy = dyn_cast<ShapedType>(zpTy)) {
      if (!shapedTy.hasStaticShape()) {
        return rewriter.notifyMatchFailure(
            loc, "expected zero point to have static shape");
      }
    } else {
      return rewriter.notifyMatchFailure(
          loc, "expected zero point to be none or have tensor type");
    }

    if (auto scaleTy = cast<ShapedType>(adaptor.getXScale().getType());
        !scaleTy.hasStaticShape()) {
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

    // Dequantization formula is (x - zero_point) * scale
    // Cast into the destination type first

    // Cast the operands of (x - zero_point) to float32 to avoid underflows
    Type arithType = rewriter.getF32Type();
    Value casted = tosaBuilder.castToNewTensorElementType(x, arithType);
    if (zeroPoint) {
      auto zpConst = tosa::expandShape(
          rewriter, loc, zeroPoint, axis, resultType.getRank());
      Value zpConstCast =
          tosaBuilder.castToNewTensorElementType(zpConst, arithType);
      casted = tosa::CreateOpAndInfer<mlir::tosa::SubOp>(
          rewriter, loc, casted.getType(), casted, zpConstCast)
                   .getResult();
    }
    auto scaleFactorConst = tosa::expandShape(
        rewriter, loc, adaptor.getXScale(), axis, resultType.getRank());
    Value scaleFactorCast =
        tosaBuilder.castToNewTensorElementType(scaleFactorConst, arithType);
    Value mulOp = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(
        rewriter, loc, casted.getType(), casted, scaleFactorCast, 0)
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
