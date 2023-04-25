/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXQuantizeLinearOp.cpp - ONNXQuantizeLinearOp --------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// ==================================================================================
//
// This file lowers ONNXQuantizeLinearOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXQuantizeLinearOpLoweringToTOSA : public OpConversionPattern<ONNXQuantizeLinearOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXQuantizeLinearOp op, OpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value x = op.x();
    Type xType = x.getType();
    ArrayRef<int64_t> inputShape =  cast<TensorType>(xType).getShape();
    Value y_scale = op.y_scale();
    Value y_zero_point = op.y_zero_point();
    // Axis attribute is ignored for per-tensor quantization, which is the only one handled
    // for the moment, so there is no need to look at this attribute.
    // If y_scale is an array, it means it is trying to run per element quantization,
    // which is not supported.
    if (cast<TensorType>(y_scale.getType()).getRank() >= 1) {
      return rewriter.notifyMatchFailure(
          op, "Only per-tensor quantization is handled.");
    }
    
    // Since tosa.add doesn't allow different shapes, get the value from the zero point
    // constant, and create a constant of the same shape as the input out of it in order
    // to have a correct add.
    mlir::ElementsAttr zeroPoint;
    if (auto source = y_zero_point.getDefiningOp<ONNXConstantOp>()) {
      zeroPoint = source.value().value(); 
    }
    else if (y_zero_point.getDefiningOp<mlir::tosa::ConstOp>()) {
      zeroPoint = tosa::getValueFromTosaConst<ElementsAttr>(y_zero_point);
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
    // Replace the division by a reciprocal followed by a mul
    Value recOp = tosa::CreateOpAndInfer<mlir::tosa::ReciprocalOp>(rewriter, loc, xType, x).getResult();
    Value mulOp = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(rewriter, loc, xType, recOp, y_scale, 0).getResult();
    Value addOp = tosa::CreateOpAndInfer<mlir::tosa::AddOp>(rewriter, loc, xType, mulOp, zpConst).getResult();
    // Cast into the result type
    Value castOp = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(rewriter, loc, op.getResult().getType(), addOp).getResult();

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXQuantizeLinearOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXQuantizeLinearOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir