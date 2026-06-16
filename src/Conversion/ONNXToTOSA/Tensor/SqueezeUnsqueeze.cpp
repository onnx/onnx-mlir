/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- SqueezeUnsqueeze.cpp - Squeeze/Unsqueeze Op ----------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX Squeeze and Unsqueeze operators to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

template <typename ONNXOpType>
class ONNXSqueezeUnsqueezeOpLoweringToTOSA
    : public OpConversionPattern<ONNXOpType> {
public:
  using OpConversionPattern<ONNXOpType>::OpConversionPattern;
  using OpAdaptor = typename ONNXOpType::Adaptor;
  LogicalResult matchAndRewrite(ONNXOpType op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TosaBuilder tosaBuilder(rewriter, op.getLoc());

    Type outputTy = op.getResult().getType();
    if (!hasStaticShape(outputTy))
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");

    Value data = op.getData();
    Value reshapeOp = tosaBuilder.reshape(
        data, mlir::cast<RankedTensorType>(outputTy).getShape());
    rewriter.replaceOp(op, {reshapeOp});
    return success();
  }
};

} // namespace

void populateLoweringONNXSqueezeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXSqueezeUnsqueezeOpLoweringToTOSA<ONNXSqueezeOp>>(
      typeConverter, ctx);
}

void populateLoweringONNXUnsqueezeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXSqueezeUnsqueezeOpLoweringToTOSA<ONNXUnsqueezeOp>>(
      typeConverter, ctx);
}

} // namespace onnx_mlir
