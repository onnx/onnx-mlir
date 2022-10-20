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

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

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
    ONNXShapeOpShapeHelper shapeHelper(&shapeOp);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    Type outputType = *op->result_type_begin();
    assert(outputType.isa<ShapedType>() && "Expected ShapedType");
    ShapedType outputShapedType = outputType.cast<ShapedType>();
    Type elementType = outputShapedType.getElementType();
    Type resultOutputType = RankedTensorType::get(
        shapeHelper.dimsForOutput(0)[0].getLiteral(), elementType);

    Value input = shapeOp.data();
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
