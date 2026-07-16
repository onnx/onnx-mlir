/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Concat.cpp - Concat Op ---------------------------===//
//
// Copyright (c) 2026 TIER IV, Inc.
//
// =============================================================================
//
// This file lowers ONNX concat operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXConcatOpLoweringToTOSA : public OpConversionPattern<ONNXConcatOp> {
public:
  using OpConversionPattern<ONNXConcatOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXConcatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto resultTensorType = mlir::dyn_cast<RankedTensorType>(resultType);
    if (!resultTensorType)
      return rewriter.notifyMatchFailure(op, "result is not a ranked tensor");

    int64_t rank = resultTensorType.getRank();
    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis is out of range");

    tosa::CreateReplaceOpAndInfer<mlir::tosa::ConcatOp>(
        rewriter, op, resultType, adaptor.getInputs(), axis);
    return success();
  }
};

} // namespace

void populateLoweringONNXConcatOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXConcatOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
