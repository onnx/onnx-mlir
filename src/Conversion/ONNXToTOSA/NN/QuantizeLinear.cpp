/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXQuantizeLinearOp.cpp - ONNXQuantizeLinearOp --------------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include "llvm/ADT/SmallVector.h"
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
    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    Value x = op.x();
    Type xType = x.getType();
    ArrayRef<int64_t> inputShape =  cast<TensorType>(xType).getShape();
    Value y_scale = op.y_scale();
    Value y_zero_point = op.y_zero_point();
    Type resultType = op.getResult().getType();
    // Axis attribute is ignored for per-tensor quantization, which is the only one handled
    // for the moment, so there is no need to look at this attribute.
    // If y_scale is an array, it means it is trying to run per element quantization,
    // which is not supported.
    if (cast<TensorType>(y_scale.getType()).getRank() >= 1) {
      return rewriter.notifyMatchFailure(
          op, "Only per-tensor quantization is handled.");
    }
    
    // Since tosa.add and tosa.mul don't allow different ranks, get the value from the
    // constants, and create a new constant of the same rank as the input out of it in order
    // to have a correct add and mul.
    mlir::ElementsAttr zeroPoint = tosa::getElementsAttrFromConst(y_zero_point);
    auto zpValue = zeroPoint.getValues<int8_t>()[0];
    auto zpConst = tosaBuilder.getSplattedConst<int8_t>(zpValue, inputShape.size());
    mlir::ElementsAttr scaleFactor = tosa::getElementsAttrFromConst(y_scale);
    auto scaleFactorValue = scaleFactor.getValues<float>()[0];
    auto scaleFactorConst = tosaBuilder.getSplattedConst<float>(scaleFactorValue, inputShape.size());
    
    // Quantization formula is ((x / y_scale) + y_zero_point)
    // Replace the division by a reciprocal followed by a mul
    Value recOp = tosa::CreateOpAndInfer<mlir::tosa::ReciprocalOp>(rewriter, loc, xType, x).getResult();
    Value mulOp = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(rewriter, loc, xType, recOp, scaleFactorConst, 0).getResult();
    // Cast into the result type
    Value castOp = tosa::CreateOpAndInfer<mlir::tosa::CastOp>(rewriter, loc, resultType, mulOp).getResult();
    Value addOp = tosa::CreateOpAndInfer<mlir::tosa::AddOp>(rewriter, loc, resultType, castOp, zpConst).getResult();

    rewriter.replaceOp(op, addOp);
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