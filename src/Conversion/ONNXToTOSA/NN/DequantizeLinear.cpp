/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXDequantizeLinearOp.cpp - ONNXDequantizeLinearOp --------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// ======================================================================================
//
// This file lowers ONNXDequantizeLinearOp operator to TOSA dialect.
//
//===--------------------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXDequantizeLinearOpLoweringToTOSA : public OpConversionPattern<ONNXDequantizeLinearOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXDequantizeLinearOp op, OpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value x = op.x();
    Value x_scale = op.x_scale();
    Value x_zero_point = op.x_zero_point();
    Type resultType = op.getResult().getType();
    // Axis attribute is ignored for per-tensor quantization, which is the only one handled
    // for the moment, so there is no need to look at this attribute.
    // If x_scale is an array, it means it is trying to run per element quantization,
    // which is not supported.
    if (cast<TensorType>(x_scale.getType()).getRank() >= 1) {
      return rewriter.notifyMatchFailure(
          op, "Only per-tensor quantization is handled.");
    }

    // Dequantization formula is (x - zero_point) * scale
    // Cast into the destination type first
    Value castOp = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(rewriter, loc, resultType, x).getResult();
    Value subOp = tosa::CreateOpAndInfer<mlir::tosa::SubOp>(rewriter, loc, resultType, castOp, x_zero_point).getResult();
    Value mulOp = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(rewriter, loc, resultType, subOp, x_scale, 0).getResult();

    rewriter.replaceOp(op, mulOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXDequantizeLinearOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXDequantizeLinearOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir