/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Conv2D.cpp - Conv2D Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX conv operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXConvOpLoweringToTOSA : public OpConversionPattern<ONNXConvOp> {
public:
  using OpConversionPattern<ONNXConvOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXConvOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXConvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.X();
    auto weights = adaptor.W();
    auto bias = adaptor.B();

    auto weightType = weights.getType().cast<ShapedType>();

    auto weightShape = weightType.getShape();

    StringAttr autopad = adaptor.auto_padAttr();
    ArrayAttr dilations = adaptor.dilationsAttr();
    IntegerAttr group = adaptor.groupAttr();
    ArrayAttr pads = adaptor.padsAttr();
    ArrayAttr strides = adaptor.stridesAttr();

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    if (group.getSInt() != 1) {
      return rewriter.notifyMatchFailure(
          op, "ONNX Conv grouping not supported");
    }

    // NOTE: we would like if inferShapes() had filled in explicit padding
    // but currently inferShapes() does not do this for ConvOp (it does for
    // ConvTransposeOp). We have not implemented code for autopad so fail.
    if (autopad && autopad != "NOTSET")
      return op.emitError("padding must be explicit");

    // Convert input [N,C,H,W] -> [N,H,W,C]
    Value newInput =
        tosa::createTosaTransposedTensor(rewriter, op, input, {0, 2, 3, 1});

    // Convert weights [M,C,H,W] -> [M,H,W,C]
    Value newWeight =
        tosa::createTosaTransposedTensor(rewriter, op, weights, {0, 2, 3, 1});

    if (bias.getType().isa<NoneType>()) {
      DenseElementsAttr newBiasAttr = DenseElementsAttr::get(
          RankedTensorType::get({weightShape[0]}, rewriter.getF32Type()),
          {0.0F});
      bias = rewriter.create<tosa::ConstOp>(
          op->getLoc(), newBiasAttr.getType(), newBiasAttr);
    }
    if (!dilations) {
      dilations = rewriter.getI64ArrayAttr({1, 1});
    }
    if (!strides) {
      strides = rewriter.getI64ArrayAttr({1, 1});
    }
    if (!pads) {
      pads = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    } else {
      llvm::SmallVector<int64_t, 4> newPadVec = extractFromI64ArrayAttr(pads);
      pads = rewriter.getI64ArrayAttr(
          {newPadVec[0], newPadVec[2], newPadVec[1], newPadVec[3]});
    }

    Type newConvOutputType = RankedTensorType::get(
        {-1, -1, -1, -1}, resultType.cast<ShapedType>().getElementType());

    Value conv2D = tosa::CreateOpAndInfer<tosa::Conv2DOp>(rewriter,
        op->getLoc(), newConvOutputType, newInput, newWeight, bias, pads,
        strides, dilations);

    // Convert output [N,H,W,M] -> [N,M,H,W]
    Value newOutput =
        tosa::createTosaTransposedTensor(rewriter, op, conv2D, {0, 3, 1, 2});

    rewriter.replaceOp(op, {newOutput});
    return success();
  }
};
} // namespace

void populateLoweringONNXConvOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir