/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Torch dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;

namespace onnx_mlir {

namespace {

// AtenAddOp requires an additional alpha parameter and thus requires a unique
// lowering
class ConvertONNXAddOp : public OpConversionPattern<ONNXAddOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value one = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    auto newResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<Torch::AtenAddTensorOp>(
        op, newResultType, adaptor.A(), adaptor.B(), one);
    return success();
  }
};
} // namespace

void populateLoweringONNXElementwiseOpToTorchPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    MLIRContext *ctx) {
  patterns.add<ConvertONNXAddOp>(typeConverter, ctx);
}

} // namespace onnx_mlir
