/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ReduceMean.cpp - ONNX Op Transform ---------------------===//
//
// Copyright 2022, Helprack LLC.
//
// =============================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;

//
// ONNX ReduceMean operation
//
// Computes the mean of the input tensor’s element along the provided axes.
// The output tensor has the same rank as the input if keepdims equal 1.
// If keepdims equal 0, then the resulted tensor have the reduced
// dimension pruned. The above behavior is similar to numpy, with the
// exception that numpy default keepdims to False instead of True.
//
// Attributes:
//   axes	::mlir::ArrayAttr	64-bit integer array attribute
//   keepdims	::mlir::IntegerAttr	64-bit signed integer attribute
//
//  Operands:
//    data	tensor of 32-bit/64-bit unsigned integer values or
//  		tensor of 32-bit/64-bit signless integer values or
//  		tensor of 16-bit/32-bit/64-bit float values or
//  		tensor of bfloat16 type values or memref of any type values.
//
//  Results:
//    reduced	tensor of 32-bit/64-bit unsigned integer values or
//  		tensor of 32-bit/64-bit signless integer values or
//  		tensor of 16-bit/32-bit/64-bit float values or tensor
//  		of bfloat16 type values or memref of any type values
//
class ONNXReduceMeanOpToTorchLowering
    : public OpConversionPattern<ONNXReduceMeanOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReduceMeanOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto axis = mlir::extractFromI64ArrayAttr(op.axesAttr());
    if (!(axis.size() == 2 && axis[0] == 2 && axis[1] == 3))
      op.emitError("Not implemented yet for general axis sizes");

    // TODO: Fully support keep dimensions attribute
    Value keepDimsVal = rewriter.create<Torch::ConstantBoolOp>(loc, true);

    IntegerType sintType = rewriter.getIntegerType(64, true);
    std::vector<Value> axisVal =
        createArrayAttribute(op.axesAttr(), sintType, loc, rewriter, 1);
    Value axisList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{axisVal});

    Value dtype = rewriter.create<ConstantNoneOp>(loc);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto newOp = rewriter.replaceOpWithNewOp<Torch::AtenMeanDimOp>(
        op, resultType, adaptor.data(), axisList, keepDimsVal, dtype);
    setLayerNameAttr(op, newOp);
    return success();
  }
};

void populateLoweringONNXToTorchReduceMeanOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXReduceMeanOpToTorchLowering>(typeConverter, ctx);
}
