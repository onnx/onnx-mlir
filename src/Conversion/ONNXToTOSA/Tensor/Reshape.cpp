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

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXReshapeOpLoweringToTOSA : public OpConversionPattern<ONNXReshapeOp> {
public:
  using OpConversionPattern<ONNXReshapeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TosaBuilder tosaBuilder(rewriter, op.getLoc());

    RankedTensorType outputTy = op.getType().template cast<RankedTensorType>();
    if (llvm::any_of(outputTy.getShape(), ShapedType::isDynamic))
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");

    Value data = op.getData();
    auto tosaReshapeOp = tosaBuilder.reshape(data, outputTy.getShape());
    rewriter.replaceOp(op, {tosaReshapeOp});
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
