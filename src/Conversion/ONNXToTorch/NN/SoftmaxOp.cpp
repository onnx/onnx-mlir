/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- SoftmaxOp.cpp - ONNX Op Transform ------------------===//
//
// =======================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;


/**
 * 
 * ONNX Softmax operation 
 *
 * “The operator computes the normalized exponential values for the given
 * input:” “” “ Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input),
 * axis=axis, keepdims=1) “ “” “The input does not need to explicitly be a
 * 2D vector. The "axis" attribute” “indicates the dimension along which 
 * Softmax will be performed.” “The output tensor has the same shape” “and
 * contains the Softmax values of the corresponding input.”
 *
 * Attributes:
 * 	axis	::mlir::IntegerAttr	64-bit signed integer attribute
 *
 *  ONNX Axis attribute Value is map to dimension in torch side.
 *
 * Operands :
 *   input	tensor of 16-bit/32-bit/64-bit float values or
 *   		tensor of bfloat16 type values or memref of any type values
 *   ONNX input is map to input in the torch side.
 *
 * Output   : 
 *   output   	tensor of 16-bit/32-bit/64-bit float values or              
 *              tensor of bfloat16 type values or memref of any type values
 *
 */


class ONNXSoftmaxOpToTorchLowering : public ConversionPattern {
public:
  ONNXSoftmaxOpToTorchLowering(TypeConverter &typeConverter, 
	MLIRContext *ctx)
      : ConversionPattern( typeConverter,
	mlir::ONNXSoftmaxOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context =  op->getContext();
    ONNXSoftmaxOp op1 = llvm::dyn_cast<ONNXSoftmaxOp>(op);
    ONNXSoftmaxOpAdaptor adapter(op1);

    auto axis = op1.axisAttr();       // ::mlir::IntegerAttr
    Value input = op1.input();

    auto inputTy = toTorchType (context, input.getType());
    auto inputTorchTensor  = 
	rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( 
			loc, inputTy, input);
    Value constAxisValue = rewriter.create<ConstantIntOp>(loc,axis);
    auto resultTy = toTorchType (context, op->getResult(0).getType());
    Value floatType = rewriter.create<ConstantBoolOp>(loc, true);
    Value result = rewriter.create<Aten_SoftmaxOp>(loc, resultTy, 
		    inputTorchTensor, constAxisValue, floatType);

    llvm::outs() << "ATENSOFTMAX CREATED \n" << result << "\n\n"; 
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, 
		    resultTy, result);
    return success();
  }
};

void populateLoweringONNXToTorchSoftmaxOpPattern(RewritePatternSet 
	&patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXSoftmaxOpToTorchLowering>(typeConverter, ctx);
}



