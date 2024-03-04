/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Dim.cpp - Lowering Dim Op ----------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNXDim operator to the Tensor dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXDimOpLoweringToStablehlo : public ConversionPattern {
  ONNXDimOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ONNXDimOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXDimOp dimOp = cast<ONNXDimOp>(op);
    int64_t axis = dimOp.getAxis();

    // Check that axis is a valid dimension index
    Value tensorArg = operands[0];
    if (!tensorArg.getType().isa<RankedTensorType>()) {
      return rewriter.notifyMatchFailure(op, "Expected ranked tensor type");
    }
    RankedTensorType tensorType = tensorArg.getType().cast<RankedTensorType>();
    int64_t rank = tensorType.getRank();

    if (axis < 0 || axis >= rank) {
      return rewriter.notifyMatchFailure(
          op, "Invalid axis, must be in range 0 to rank-1 of the input tensor");
    }
    Value dimValue = rewriter.create<tensor::DimOp>(loc, tensorArg, axis);

    Type dimType = dimOp.getDim().getType();
    Type indexValueType = dimType.cast<ShapedType>().getElementType();
    Value castedIndex =
        rewriter.create<arith::IndexCastOp>(loc, indexValueType, dimValue);
    Value indexTensor = rewriter.create<tensor::FromElementsOp>(
        loc, dimType, ArrayRef<Value>{castedIndex});
    rewriter.replaceOp(op, indexTensor);
    return success();
  }
};

} // namespace

void populateLoweringONNXDimOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXDimOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
