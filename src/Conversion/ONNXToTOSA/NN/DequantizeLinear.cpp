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
    ArrayRef<int64_t> inputShape =  cast<TensorType>(x.getType()).getShape();
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

    // Since tosa.sub doesn't allow different shapes, get the value from the zero point
    // constant, and create a constant of the same shape as the input out of it in order
    // to have a correct sub.
    mlir::ElementsAttr zeroPoint;
    if (auto source = x_zero_point.getDefiningOp<ONNXConstantOp>()) {
      zeroPoint = source.value().value(); 
    }
    else if (x_zero_point.getDefiningOp<mlir::tosa::ConstOp>()) {
      zeroPoint = tosa::getValueFromTosaConst<ElementsAttr>(x_zero_point);
    }
    auto zpValue = zeroPoint.getValues<APInt>()[0];
    
    uint64_t numElements = 1;
    for (int64_t a : inputShape) {
      numElements *= a;
    }
    std::vector zpVec = std::vector<APInt>(numElements, zpValue);
    auto constType =
        RankedTensorType::get(inputShape, rewriter.getIntegerType(sizeof(int8_t) * 8));
    auto constAttr = DenseElementsAttr::get(constType, zpVec);
    auto zpConst = tosa::CreateOpAndInfer<mlir::tosa::ConstOp>(rewriter, loc, constType, constAttr);
    
    // Quantization formula is ((x / y_scale) + y_zero_point)
    // Dequantization formula is (x - zero_point) * scale
    // Cast into the destination type first
    Value castOp = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(rewriter, loc, resultType, x).getResult();
    Value subOp = tosa::CreateOpAndInfer<mlir::tosa::SubOp>(rewriter, loc, resultType, castOp, zpConst).getResult();
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