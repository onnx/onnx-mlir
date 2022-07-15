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

using namespace mlir;
using namespace mlir::torch;

// Wrapper function for emit the Flatten Operation.
static Value createAtenFlattenOp(ConversionPatternRewriter &rewriter,
    Location loc, Value result, ValueTensorType resultType, int64_t start_dim,
    int64_t end_dim, ONNXFlattenOp op) {
  IntegerType ty = IntegerType::get(op.getContext(), 64);
  IntegerAttr startDimInt = IntegerAttr::get(ty, (start_dim));
  Value startDimConstInt =
      rewriter.create<Torch::ConstantIntOp>(loc, startDimInt);
  IntegerAttr endDimInt = IntegerAttr::get(ty, (end_dim));
  Value endDimConstInt = rewriter.create<Torch::ConstantIntOp>(loc, endDimInt);
  return rewriter.create<Torch::AtenFlattenUsingIntsOp>(
      loc, resultType, result, startDimConstInt, endDimConstInt);
}

// Flattens input by reshaping it into a one-dimensional tensor.
// If start_dim or end_dim are passed, only dimensions starting with
// start_dim and ending with end_dim are flattened.
// The order of elements in input is unchanged.
//
// Attributes
//    axis    	::mlir::IntegerAttr 	i64-bit signed integer attribute
//    In torch side, Calculate Start dim and End dim using this
//    axis attribute value.
//
// Operands:
//    input     tensor of 8-bit/16-bit/32-bit unsigned integer values or
//    		tensor of 64-bit unsigned integer values or
//    	  tensor of 8-bit/16-bit/32-bit/64-bit signless integer values or
//    		tensor of bfloat16 type or tensor of 16-bit float values
//    		tensor of 32-bit float or tensor of 64-bit float values
//    		tensor of string type values or tensor of 1-bit signless
//    		integer values or tensor of complex type with 32-bit float
//    		elements values or tensor of complex type with 64-bit float
//    		elements values or memref of any type values
//    Map this input operand into input parameter in torch side.
// Results:
//    output    tensor of 8-bit/16-bit/32-bit unsigned integer values or
//              tensor of 64-bit unsigned integer values or
//              tensor of 8-bit/16-bit/32-bit/64-bit signless integer
//              values or
//              tensor of bfloat16 type or tensor of 16-bit float values
//              tensor of 32-bit float or tensor of 64-bit float values
//              tensor of string type values or tensor of 1-bit signless
//              integer values or tensor of complex type with 32-bit float
//              elements values or tensor of complex type with 64-bit float
//              elements values or memref of any type values
//
class ONNXFlattenOpToTorchLowering : public OpConversionPattern<ONNXFlattenOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXFlattenOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    Value input = op.input();
    int64_t axisValue = op.axis();

    TensorType resultTensorType = op.getResult().getType().cast<TensorType>();
    auto resultType = Torch::ValueTensorType::get(context,
        resultTensorType.getShape(), resultTensorType.getElementType());

    TensorType inputTensorType = input.getType().cast<TensorType>();
    Value result = adaptor.input();

    // if axisValue is negative
    if (axisValue < 0)
      return op->emitError("negative axis not supported");

    /***************************************************************/
    // if axisValue is 0 or 1, we need to flatten once.
    // i) if axisValue is 0, start_dim will be -1 (axisValue -1 ) and
    //    end_dim will be 0.
    // ii) if axisValue is 1, start_dim will be 0 (axisValue - 1) and
    //    end_dim will 0. Flatten from 0 to 0  flatten was not emitted.
    //
    // Because of these reasons, need to emit flatten once with
    //   start=1, end=-1.
    //
    // If axisValue is more than 1, emit the flatten two times like below.
    //    a) Flattening is about bringing the flattened zone into a
    //       single dimensional.
    //    b) torch flatten flattens the region between (start-dim
    //       and end-dim).
    //    c) onnx flattening is about creating a 2-D vector,
    //       the first dim consisting of values till axis-1, the second
    //       dim consisting of values from axis to end.
    //
    // What we feel is that there will be two steps required -
    //
    //  1) flatten the region from 0 position to axis - 1.
    //  2) Since all dimensions before `axis` have already been
    //     condensed into a single one (dim 0),
    //     we set start=1. We use -1 as the end value, which tells torch
    //     to go until the last dimension.
    /********************************************************************/

    if (axisValue > 1) {
      // Build the intermediate result type.
      // This is the same type as the input, with all dims before the axis
      // value collapsed into one.
      ArrayRef<int64_t> inputShape = inputTensorType.getShape();
      int64_t numDimsAfterAxis = inputShape.size() - axisValue;
      auto remainingDims = inputShape.take_back(numDimsAfterAxis);
      std::vector<int64_t> intermShape;
      // this is the collapsed dimension
      intermShape.push_back(resultTensorType.getShape()[0]);
      intermShape.insert(
          intermShape.end(), remainingDims.begin(), remainingDims.end());
      auto intermType = Torch::ValueTensorType::get(context,
          llvm::makeArrayRef(intermShape), inputTensorType.getElementType());
      // 1) Flatten the region from 0 position to axis - 1.
      result = createAtenFlattenOp(rewriter, loc, result, intermType,
          /*start=*/0, /*start=*/axisValue - 1, op);
    }

    // 2) Flatten the region from start=1, end=-1.
    result = createAtenFlattenOp(rewriter, loc, result, resultType,
        /*start=*/1, /*end=*/-1, op);
    rewriter.replaceOpWithNewOp<Torch::TensorStaticInfoCastOp>(
        op, resultType, result);
    return success();
  }
};

void populateLoweringONNXToTorchFlattenOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXFlattenOpToTorchLowering>(typeConverter, ctx);
}
