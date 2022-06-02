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

struct ONNXAddOpToTorchLowering : public ConversionPattern {

  Value getAlphaDefaultValue(MLIRContext *context,
      ConversionPatternRewriter &rewriter, Location loc) const {
    auto I64type = IntegerType::get(context, 64);
    auto one = 1;
    auto oneIntAttr = IntegerAttr::get(I64type, one);
    return rewriter.create<ConstantIntOp>(loc, oneIntAttr);
  }

  ONNXAddOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXAddOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXAddOp addOp = llvm::dyn_cast_or_null<ONNXAddOp>(op);

    assert(addOp && "Expecting op to have a strong type");

    Location loc = addOp.getLoc();
    mlir::MLIRContext *context = addOp.getContext();

    Value alphaDefaultValue = getAlphaDefaultValue(context, rewriter, loc);
    auto aTensor = getTorchTensor(addOp.A(), rewriter, context, loc);
    auto bTensor = getTorchTensor(addOp.B(), rewriter, context, loc);

    auto resultType = toTorchType(context, addOp.getResult().getType());
    Value result = rewriter.create<AtenAddTensorOp>(
        loc, resultType, aTensor, bTensor, alphaDefaultValue);

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);
    return success();
  }
};

void populateLoweringONNXToTorchAddOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXAddOpToTorchLowering>(typeConverter, ctx);
}
