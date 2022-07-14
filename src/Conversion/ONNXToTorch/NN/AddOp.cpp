/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- AddOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// ========================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===-----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/**
 *
 * ONNX Add operation
 *
 * “Performs element-wise binary addition (with Numpy-style broadcasting
 * support).” “” “This operator supports multidirectional
 * (i.e. Numpy-style) broadcasting; for more details please check the doc.”
 *
 * Operands :
 *    A	  tensor of 32-bit/64-bit unsigned integer values or
 *        tensor of 32-bit/64-bit signless integer values or
 *        tensor of 16-bit/32-bit/64-bit float values or
 *        tensor of bfloat16 type values or memref of any type values
 *  Map this A operand with input parameter in torch side.
 *    B   tensor of 32-bit/64-bit unsigned integer values or
 *        tensor of 32-bit/64-bit signless integer values or
 *        tensor of 16-bit/32-bit/64-bit float values or
 *        tensor of bfloat16 type values or memref of any type values
 *  Map this B operand with other parameter in torch side.
 *
 * Results:
 *   C    tensor of 32-bit/64-bit unsigned integer values or
 *        tensor of 32-bit/64-bit signless integer values or
 *        tensor of 16-bit/32-bit/64-bit float values or
 *        tensor of bfloat16 type values or memref of any type values
 *
 */

struct ONNXAddOpToTorchLowering : public OpConversionPattern<ONNXAddOp> {

  static Value getAlphaDefaultValue(MLIRContext *context,
      ConversionPatternRewriter &rewriter, Location loc)  {
    auto I64type = IntegerType::get(context, 64);
    auto oneIntAttr = IntegerAttr::get(I64type, 1);
    return rewriter.create<ConstantIntOp>(loc, oneIntAttr);
  }

  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

    Value alphaDefaultValue = getAlphaDefaultValue(context, rewriter, loc);
    Value aTensor = getTorchTensor(op.A(), rewriter, context, loc);
    Value bTensor = getTorchTensor(op.B(), rewriter, context, loc);

    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    Value result = rewriter.create<AtenAddTensorOp>(
        loc, resultType, aTensor, bTensor, alphaDefaultValue);

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, resultType, result);
    return success();
  }
};

void populateLoweringONNXToTorchAddOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXAddOpToTorchLowering>(typeConverter, ctx);
}
