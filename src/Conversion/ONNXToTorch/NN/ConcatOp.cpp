/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ConcatOp.cpp - ONNX Op Transform -----------------------===//
//
// =======================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

#ifdef _WIN32
#include <io.h>
#endif

/*
 * “Concatenate a list of tensors into a single tensor. 
 * All input tensors must have the same shape, except for the dimension 
 * size of the axis to concatenate on.”
 *
 * Attributes:
 *	axis	::mlir::IntegerAttr	64-bit signed integer attribute
 *  ONNX axis value is map to dimension in the torch side.
 *
 * Operands:
 *    inputs	tensor of 8-bit/16-bit/32-bit/64-bit unsigned 
 *    		integer values or tensor of 8-bit/16-bit/32-bit/64-bit 
 *    		signless integer values or tensor of bfloat16 type values 
 *    		or tensor of 16-bit/32-bit/64-bit float values or 
 *    		tensor of string type values or tensor of 1-bit signless 
 *    		integer values or tensor of complex type with 32-bit/64-bit
 *    		float elements values or memref of any type values.
 *    ONNX inputs map to input tensors in torch side.
 *
 * Results:
 * concat_result    tensor of 8-bit/16-bit/32-bit/64-bit unsigned
 *              integer values or tensor of 8-bit/16-bit/32-bit/64-bit
 *              signless integer values or tensor of bfloat16 type values
 *              or tensor of 16-bit/32-bit/64-bit float values or
 *              tensor of string type values or tensor of 1-bit signless
 *              integer values or tensor of complex type with 32-bit/64-bit
 *              float elements values or memref of any type values.
 */
using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;


class ONNXConcatOpToTorchLowering : public ConversionPattern {
public:
  ONNXConcatOpToTorchLowering(TypeConverter &typeConverter, 
	MLIRContext *ctx)
      : ConversionPattern(
	typeConverter, ::mlir::ONNXConcatOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context =  op->getContext();
    ONNXConcatOp op1 = llvm::dyn_cast<ONNXConcatOp>(op);
    ONNXConcatOpAdaptor adaptor(op1);
    
    ValueRange inputs = op1.inputs();
    auto axisValue = op1.axisAttr();       // ::mlir::IntegerAttr
    Value  axisVal = rewriter.create<ConstantIntOp>(loc,axisValue);
    
    auto resultType = toTorchType(context, op1.getType());
    std::vector<Value> inputArrayValues;
    // iterate through the list of inputs and create the 
    // Torch Tensors of each input.
    for (unsigned int i = 0; i < inputs.size(); i++)
    {
      auto inputType = toTorchType(context, inputs[i].getType());
      auto inputTorchTensor  = 
	rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
		      loc, inputType, inputs[i]);
      inputArrayValues.push_back(inputTorchTensor);
    }
    // here, resultType is different than input tensor type.
    // here, we are creting list of input tensor types. And all 
    // input tensor list types will be the same. That's why i have taken
    // one element type from that list(inputArrayValues.front().getType()).
    // All element types will be same
    // from that list right?
    // E.g:
    //    %2 = "torch.prim.ListConstruct"(%arg0, %arg0) : 
    //     (!torch.vtensor<[5,5],f32>, !torch.vtensor<[5,5],f32>) -> 
    //     !torch.list<vtensor<[5,5],f32>>

    Value inputShapeList = rewriter.create<PrimListConstructOp>(loc, 
	Torch::ListType::get(inputArrayValues.front().getType()),
       			ValueRange{inputArrayValues});

    Value result = rewriter.create<AtenCatOp>(loc, resultType, 
		    inputShapeList, axisVal);
    
    llvm::outs() << "Aten Concat Op:   " << "\n" << result 
	    << "\n" << "\n";
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType,
		    result);
    return success();
  }
};

void populateLoweringONNXToTorchConcatOpPattern(RewritePatternSet 
	&patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXConcatOpToTorchLowering>(typeConverter, ctx);
}
