/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Reshape.cpp - Reshape Op ----------------------------===//
//
// Copyright (c) d-Matrix Inc. 2023
//
// =============================================================================
//
// This file lowers ONNX reshape operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXReshapeOpLoweringToTOSA : public OpConversionPattern<ONNXReshapeOp> {
public:
  using OpConversionPattern<ONNXReshapeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getAllowzero())
      return rewriter.notifyMatchFailure(
          op, "lowering with `allowzero = 1` attribute not supported");

    TosaBuilder tosaBuilder(rewriter, op.getLoc());

    Type outputTy = op.getType();
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

void populateLoweringONNXReshapeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
