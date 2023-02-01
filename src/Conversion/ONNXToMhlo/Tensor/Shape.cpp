/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Shape.cpp - Lowering Shape Op ----------------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Shape Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXShapeOpLoweringToMhlo : public ConversionPattern {
  ONNXShapeOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXShapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get shape.
    ONNXShapeOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    ONNXShapeOp shapeOp = cast<ONNXShapeOp>(op);
    Location loc = op->getLoc();
    IndexExprBuilderForMhlo createIE(rewriter, loc);
    ONNXShapeOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Type outputType = *op->result_type_begin();
    assert(outputType.isa<ShapedType>() && "Expected ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
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

void populateLoweringONNXShapeOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXShapeOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
