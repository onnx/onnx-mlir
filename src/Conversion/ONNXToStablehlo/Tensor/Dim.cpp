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

#include "mlir/Dialect/Shape/IR/Shape.h"
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
    int64_t axisLit = dimOp.getAxis();

    // Check that axisLit is a valid dimension index
    Value tensorArg = operands[0];
    assert(mlir::isa<RankedTensorType>(tensorArg.getType()) &&
           "Expected ranked tensor type");

    int64_t rank = mlir::cast<RankedTensorType>(tensorArg.getType()).getRank();

    assert((axisLit >= 0 && axisLit < rank) &&
           "Axis must be in the range [0, input tensor rank - 1]");

    Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, tensorArg);
    Value dimValue =
        rewriter.create<shape::GetExtentOp>(loc, inputShape, axisLit);
    Type dimType = dimOp.getDim().getType();
    Type indexValueType = mlir::cast<ShapedType>(dimType).getElementType();
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
