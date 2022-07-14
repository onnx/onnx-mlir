/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ Argmax.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// ====================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===--------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;

// ONNX Argmax operation
//
// Computes the indices of the max elements of the input tensor's element
// along the provided axis. The resulting tensor has the same rank as the
// input if keepdims equal 1. If keepdims equal 0, then the resulting
// tensor have the reduced dimension pruned. If select_last_index is True
// (default False), the index of the last occurrence of the max is
// selected if the max appears more than once in the input. Otherwise the
// index of the first occurrence is selected. The type of the output
// tensor is integer.
//
// Operands :
//  data
//    tensor of 8-bit unsigned integer values or tensor of 16-bit
//    unsigned integer values or tensor of 32-bit unsigned integer values
//    or tensor of 64-bit unsigned integer values or tensor of 8-bit
//    signless integer values or tensor of 16-bit signless integer values
//    or tensor of 32-bit signless integer values or tensor of 64-bit
//    signless integer values or tensor of 16-bit float values or tensor
//    of 32-bit float values or tensor of 64-bit float values or tensor
//    of bfloat16 type values or memref of any type values.
//
// Attributes
//       axis	::mlir::IntegerAttr	64-bit signed integer attribute
//     keepdims	  ::mlir::IntegerAttr 	64-bit signed integer attribute
// select_last_index   ::mlir::IntegerAttr	64-bit signed integer
// 								attribute
//
// Results:
//   reduced	tensor of 64-bit signless integer values or memref of
//   			any type values.
//
// ArgMax op return type is i64 signless integer, but in Torch side, don't
// have support for 64-bit signless integer values.
// Because of this reason we have implemented type conversion from 64-bit
// signless integer to 64-bit signed integer.
//
class ONNXArgMaxOpToTorchLowering : public OpConversionPattern<ONNXArgMaxOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXArgMaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    mlir::IntegerAttr axis = op.axisAttr();
    int64_t keepdims = op.keepdims();
    mlir::IntegerAttr select_last_index = op.select_last_indexAttr();

    if (select_last_index && op.select_last_index() != 0)
      return op.emitError("select_last_index is currently not supported");

    Value dim = rewriter.create<Torch::ConstantIntOp>(loc, axis);
    Type resultTy = getTypeConverter()->convertType(op->getResult(0).getType());
    Value keepDimVal;
    if (keepdims == 0)
      keepDimVal = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    else
      keepDimVal = rewriter.create<Torch::ConstantBoolOp>(loc, true);
    rewriter.replaceOpWithNewOp<Torch::AtenArgmaxOp>(
        op, resultTy, adaptor.data(), dim, keepDimVal);
    return success();
  }
};

void populateLoweringONNXToTorchArgmaxOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXArgMaxOpToTorchLowering>(typeConverter, ctx);
}
