/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Squeeze.cpp - Lowering Squeeze Op ----------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Squeeze Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXSqueezeOp(A) is implemented using Stablehlo reshapeOp
struct ONNXSqueezeOpLoweringToStablehlo : public ConversionPattern {
  ONNXSqueezeOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSqueezeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeOpAdaptor operandAdaptor(operands);
    ONNXSqueezeOp squeezeOp = llvm::cast<ONNXSqueezeOp>(op);
    Location loc = op->getLoc();
    Value data = squeezeOp.getData();
    Value axes = squeezeOp.getAxes();
    assert(isRankedShapedType(data.getType()) &&
           "data must be ranked Shaped Type");
    ShapedType dataType = mlir::cast<ShapedType>(data.getType());
    int64_t rank = dataType.getRank();

    // Shape helper is unused
    IndexExprBuilderForStablehlo createIE(rewriter, loc);
    ONNXSqueezeOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    SmallVector<int64_t, 4> axesList;
    if (ElementsAttr axesAttr = getElementAttributeFromONNXValue(axes)) {
      for (IntegerAttr value : axesAttr.getValues<IntegerAttr>()) {
        int64_t axis = mlir::cast<IntegerAttr>(value).getInt();
        if (axis < 0)
          axis += rank;
        axesList.push_back(axis);
      }
    }

    int64_t newRank = rank - axesList.size();
    SmallVector<Value, 4> newShape;
    SmallVector<bool, 4> isSqueezeDim(rank, false);
    Value dataShape = shape::ShapeOfOp::create(rewriter, loc, data);
    for (int64_t axis : axesList) {
      isSqueezeDim[axis] = true;
    }
    for (int64_t i = 0; i < rank; i++) {
      if (!isSqueezeDim[i]) {
        Value dim = shape::GetExtentOp::create(rewriter, loc, dataShape, i);
        newShape.push_back(dim);
      }
    }
    Type outputShapeType =
        RankedTensorType::get({newRank}, rewriter.getIndexType());
    Value newShapeValue = shape::FromExtentsOp::create(rewriter, loc, newShape);
    newShapeValue = shape::ToExtentTensorOp::create(
        rewriter, loc, outputShapeType, newShapeValue);
    Type outputType = *op->result_type_begin();
    Value result = stablehlo::DynamicReshapeOp::create(
        rewriter, loc, outputType, data, newShapeValue);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXSqueezeOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSqueezeOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
