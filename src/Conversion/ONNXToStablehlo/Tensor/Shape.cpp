/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Shape.cpp - Lowering Shape Op ----------------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Shape Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXShapeOpLoweringToStablehlo : public ConversionPattern {
  ONNXShapeOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXShapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get shape.
    ONNXShapeOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    ONNXShapeOp shapeOp = cast<ONNXShapeOp>(op);
    Location loc = op->getLoc();
    IndexExprBuilderForStablehlo createIE(rewriter, loc);
    ONNXShapeOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Type outputType = *op->result_type_begin();
    assert(mlir::isa<ShapedType>(outputType) && "Expected ShapedType");
    ShapedType outputShapedType = mlir::cast<ShapedType>(outputType);
    Type elementType = outputShapedType.getElementType();
    Type resultOutputType = RankedTensorType::get(
        shapeHelper.getOutputDims(0)[0].getLiteral(), elementType);

    Value input = shapeOp.getData();
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, input);
    Value castedShape =
        rewriter.create<arith::IndexCastOp>(loc, resultOutputType, shape);
    rewriter.replaceOp(op, castedShape);
    return success();
  }
};

} // namespace

void populateLoweringONNXShapeOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXShapeOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
