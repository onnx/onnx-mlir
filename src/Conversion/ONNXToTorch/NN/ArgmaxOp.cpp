/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ArgmaxOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// ====================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===--------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/**
 * 
 * ONNX Argmax operation 
 *
 * Computes the indices of the max elements of the input tensor's element
 * along the provided axis. The resulting tensor has the same rank as the
 * input if keepdims equal 1. If keepdims equal 0, then the resulting
 * tensor have the reduced dimension pruned. If select_last_index is True
 * (default False), the index of the last occurrence of the max is
 * selected if the max appears more than once in the input. Otherwise the
 * index of the first occurrence is selected. The type of the output
 * tensor is integer.
 *
 * Operands :
 *  data    
 *    tensor of 8-bit unsigned integer values or tensor of 16-bit 
 *    unsigned integer values or tensor of 32-bit unsigned integer values 
 *    or tensor of 64-bit unsigned integer values or tensor of 8-bit 
 *    signless integer values or tensor of 16-bit signless integer values 
 *    or tensor of 32-bit signless integer values or tensor of 64-bit 
 *    signless integer values or tensor of 16-bit float values or tensor
 *    of 32-bit float values or tensor of 64-bit float values or tensor
 *    of bfloat16 type values or memref of any type values.
 *
 * Attributes 
 *       axis	::mlir::IntegerAttr	64-bit signed integer attribute
 *     keepdims	  ::mlir::IntegerAttr 	64-bit signed integer attribute
 * select_last_index   ::mlir::IntegerAttr	64-bit signed integer
 * 								attribute 
 *
 * Results:
 *   reduced	tensor of 64-bit signless integer values or memref of 
 *   			any type values.
 *
 * ArgMax op return type is i64 signless integer, but in Torch side, don't
 * have support for 64-bit signless integer values.
 * Because of this reason we have implemented type conversion from 64-bit
 * signless integer to 64-bit signed integer.
 */

class ONNXArgmaxOpToTorchLowering : public ConversionPattern {
public:
  ONNXArgmaxOpToTorchLowering(TypeConverter &typeConverter, 
	MLIRContext *ctx) : ConversionPattern(
         typeConverter, mlir::ONNXArgMaxOp::getOperationName(), 1, ctx) {
	}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context =  op->getContext();
    ONNXArgMaxOp op1 = llvm::dyn_cast<ONNXArgMaxOp>(op);

    auto axis 		= op1.axisAttr();       // ::mlir::IntegerAttr
    int64_t keepdims	= op1.keepdims();	// ::mlir::IntegerAttr
    auto select_last_index = op1.select_last_indexAttr();  
    						// ::mlir::IntegerAttr
    Value data = op1.data();
    auto dataType = toTorchType(context, data.getType());
    auto dataTorchTensor  = 
	    rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
		    loc, dataType, data);
    Value dim 	  = rewriter.create<ConstantIntOp>(loc,axis);
    // type conversion from signless i64 to signed i64 type.
    auto resultTy = toSI64SignedType(context, op1.getType());
    Value keepDimVal;
    if (keepdims == 0)
      keepDimVal = rewriter.create<ConstantBoolOp>(loc, false);
    else
      keepDimVal = rewriter.create<ConstantBoolOp>(loc, true);
    AtenArgmaxOp result = rewriter.replaceOpWithNewOp<AtenArgmaxOp>(op,
		    resultTy, dataTorchTensor, dim, keepDimVal);
    return success();
  }
};

void populateLoweringONNXToTorchArgmaxOpPattern(RewritePatternSet 
	&patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXArgmaxOpToTorchLowering>(typeConverter, ctx);
}
