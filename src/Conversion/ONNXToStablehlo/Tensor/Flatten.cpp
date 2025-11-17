/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Flatten Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXFlattenOp(A) is mainly implemented using Stablehlo reshapeOp
struct ONNXFlattenOpLoweringToStablehlo : public ConversionPattern {
  ONNXFlattenOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXFlattenOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    ONNXFlattenOpAdaptor operandAdaptor(operands);
    ONNXFlattenOp flattenOp = llvm::cast<ONNXFlattenOp>(op);

    Value input = operandAdaptor.getInput();
    assert(isRankedShapedType(input.getType()) && "Expected Ranked ShapedType");
    ShapedType inputType = mlir::cast<ShapedType>(input.getType());
    int64_t rank = inputType.getRank();
    int64_t axis = flattenOp.getAxis();
    assert(axis >= -rank && axis <= rank - 1);
    axis = axis >= 0 ? axis : rank + axis;

    Value flattenDimFirst = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value inputShape = shape::ShapeOfOp::create(rewriter, loc, input);
    for (int64_t i = 0; i < axis; i++) {
      Value dim = shape::GetExtentOp::create(rewriter, loc, inputShape, i);
      flattenDimFirst =
          shape::MulOp::create(rewriter, loc, flattenDimFirst, dim);
    }
    Value flattenDimSecond = arith::ConstantIndexOp::create(rewriter, loc, 1);
    for (int64_t i = axis; i < rank; i++) {
      Value dim = shape::GetExtentOp::create(rewriter, loc, inputShape, i);
      flattenDimSecond =
          shape::MulOp::create(rewriter, loc, flattenDimSecond, dim);
    }
    SmallVector<Value> dims{flattenDimFirst, flattenDimSecond};
    Type outputShapeType = RankedTensorType::get({2}, rewriter.getIndexType());
    Value outputShape = shape::FromExtentsOp::create(rewriter, loc, dims);
    outputShape = shape::ToExtentTensorOp::create(
        rewriter, loc, outputShapeType, outputShape);
    auto result = stablehlo::DynamicReshapeOp::create(
        rewriter, loc, *op->result_type_begin(), input, outputShape);
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXFlattenOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXFlattenOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
