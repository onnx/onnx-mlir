/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- FlattenOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// ===================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===-------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/**
 * Flattens input by reshaping it into a one-dimensional tensor.
 * If start_dim or end_dim are passed, only dimensions starting with
 * start_dim and ending with end_dim are flattened.
 * The order of elements in input is unchanged.
 *
 * Attributes
 *    axis    	::mlir::IntegerAttr 	i64-bit signed integer attribute
 *    In torch side, Calculate Start dim and End dim using this 
 *    axis attribute value.
 *
 * Operands:
 *    input     tensor of 8-bit/16-bit/32-bit unsigned integer values or
 *    		tensor of 64-bit unsigned integer values or
 *    	       	tensor of 8-bit/16-bit/32-bit/64-bit signless integer 
 *    	       	values or
 *    		tensor of bfloat16 type or tensor of 16-bit float values
 *    		tensor of 32-bit float or tensor of 64-bit float values
 *    		tensor of string type values or tensor of 1-bit signless 
 *    		integer values or tensor of complex type with 32-bit float
 *    		elements values or tensor of complex type with 64-bit float
 *    		elements values or memref of any type values
 *    Map this input operand into input parameter in torch side.
 * Results:
 *    output     tensor of 8-bit/16-bit/32-bit unsigned integer values or
 *              tensor of 64-bit unsigned integer values or
 *              tensor of 8-bit/16-bit/32-bit/64-bit signless integer 
 *              values or
 *              tensor of bfloat16 type or tensor of 16-bit float values
 *              tensor of 32-bit float or tensor of 64-bit float values
 *              tensor of string type values or tensor of 1-bit signless
 *              integer values or tensor of complex type with 32-bit float
 *              elements values or tensor of complex type with 64-bit float
 *              elements values or memref of any type values
 *
 */

static Value createAtenFlattenOp(ConversionPatternRewriter &rewriter, 
	Location loc, Value result, ValueTensorType resultType,
	int64_t start_dim, int64_t end_dim, ONNXFlattenOp op1) {
  auto ty = IntegerType::get(op1.getContext(), 64);
  auto startDimInt = IntegerAttr::get(ty, (start_dim));
  Value startDimConstInt = 
	  rewriter.create<ConstantIntOp>(loc, startDimInt);
  auto endDimInt = IntegerAttr::get(ty, (end_dim));
  Value endDimConstInt = rewriter.create<ConstantIntOp>(loc, endDimInt);
  return rewriter.create<AtenFlattenUsingIntsOp>(loc,
                    resultType, result, startDimConstInt, endDimConstInt);
}

class ONNXFlattenOpToTorchLowering : public ConversionPattern {
public:
  ONNXFlattenOpToTorchLowering(TypeConverter &typeConverter,
		  MLIRContext *ctx)
      : ConversionPattern( typeConverter,
		      ::mlir::ONNXFlattenOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context = op->getContext();
    ONNXFlattenOp op1 = llvm::dyn_cast<ONNXFlattenOp>(op);

    Value input = op1.input();
    auto axisValue = op1.axis();       // ::mlir::IntegerAttr

    auto inputShape = input.getType().cast<ShapedType>().getShape();
    int64_t inputRank = inputShape.size();

    TensorType resultTensorType =
	    op->getResult(0).getType().cast<TensorType>();
    auto resultType = Torch::ValueTensorType::get(op1.getContext(),
          resultTensorType.getShape(), resultTensorType.getElementType());

    TensorType inputTensorType  = input.getType().cast<TensorType>();
    auto inputType =
	 Torch::ValueTensorType::get(context, inputTensorType.getShape(),
                    inputTensorType.getElementType());
    auto inputTensor =
	  rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(loc,
                    inputType, input);
    Value result = inputTensor;

    if (axisValue < 0)
      return op->emitError("negative axis not supported");
    
    if (axisValue > 1) {
      // flatten the region upto axis-1.
      result = createAtenFlattenOp (rewriter, loc, result, resultType, 0, 
		      axisValue - 1, op1);
      llvm::outs() << "Aten Flatten1 Op:   "
	      << "\n" << result << "\n" << "\n";
    }

    // flatten the region from axis upwards.
    result = createAtenFlattenOp (rewriter, loc, result, resultType,
		    axisValue, inputRank, op1);
    llvm::outs() << "AtenFlatten Op created" << "\n"
	    << "\n" << result << "\n" << "\n";
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op,
                    resultType, result);
    return success();
  }
};

void populateLoweringONNXToTorchFlattenOpPattern(RewritePatternSet 
    &patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXFlattenOpToTorchLowering>(typeConverter, ctx);
}
