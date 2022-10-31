/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNX Flatten Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXFlattenOp(A) is mainly implemented using MHLO reshapeOp
struct ONNXFlattenOpLoweringToMhlo : public ConversionPattern {
  ONNXFlattenOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXFlattenOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    ONNXFlattenOpAdaptor operandAdaptor(operands);
    ONNXFlattenOp flattenOp = llvm::cast<ONNXFlattenOp>(op);

    // shape helper
    IndexExprScope scope(&rewriter, loc);
    ONNXFlattenOpShapeHelper shapeHelper(&flattenOp, &scope);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(!failed(shapecomputed) && "shape helper failed");

    Value input = operandAdaptor.input();

    SmallVector<Value> dims;
    IndexExpr::getValues(shapeHelper.dimsForOutput(), dims);
    Type outputShapeType = RankedTensorType::get({2}, rewriter.getIndexType());
    Value outputShape = rewriter.create<shape::FromExtentsOp>(loc, dims);
    outputShape = rewriter.create<shape::ToExtentTensorOp>(
        loc, outputShapeType, outputShape);
    auto result = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, *op->result_type_begin(), input, outputShape);
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXFlattenOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXFlattenOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
