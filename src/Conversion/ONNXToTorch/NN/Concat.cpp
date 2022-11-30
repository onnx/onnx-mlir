/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- Concat.cpp - ONNX Op Transform -----------------------===//
//
// =======================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;

// Concatenate a list of tensors into a single tensor.
// All input tensors must have the same shape, except for the dimension
// size of the axis to concatenate on.
//
// Attributes:
//	axis	::mlir::IntegerAttr	64-bit signed integer attribute
//  ONNX axis value is map to dimension in the torch side.
//
// Operands:
//    inputs	tensor of 8-bit/16-bit/32-bit/64-bit unsigned
//    		integer values or tensor of 8-bit/16-bit/32-bit/64-bit
//    		signless integer values or tensor of bfloat16 type values
//    		or tensor of 16-bit/32-bit/64-bit float values or
//    		tensor of string type values or tensor of 1-bit signless
//    		integer values or tensor of complex type with 32-bit/64-bit
//    		float elements values or memref of any type values.
//    ONNX inputs map to input tensors in torch side.
//
// Results:
// concat_result    tensor of 8-bit/16-bit/32-bit/64-bit unsigned
//              integer values or tensor of 8-bit/16-bit/32-bit/64-bit
//              signless integer values or tensor of bfloat16 type values
//              or tensor of 16-bit/32-bit/64-bit float values or
//              tensor of string type values or tensor of 1-bit signless
//              integer values or tensor of complex type with 32-bit/64-bit
//              float elements values or memref of any type values.
//
class ONNXConcatOpToTorchLowering : public OpConversionPattern<ONNXConcatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXConcatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    ValueRange inputs = adaptor.inputs();
    IntegerAttr axisValue = op.axisAttr();
    Value axisVal = rewriter.create<ConstantIntOp>(loc, axisValue);
    Value inputShapeList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(
            Torch::ValueTensorType::getWithLeastStaticInformation(
                getContext())),
        inputs);

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto newOp = rewriter.replaceOpWithNewOp<AtenCatOp>(
        op, resultType, inputShapeList, axisVal);
    setLayerNameAttr(op, newOp);
    return success();
  }
};

void populateLoweringONNXToTorchConcatOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConcatOpToTorchLowering>(typeConverter, ctx);
}
