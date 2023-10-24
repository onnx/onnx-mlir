/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Reshape.cpp - Lowering Reshape Op -------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNX Reshape Operator to StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStableHlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStableHlo/ONNXToStableHloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXReshapeOpLoweringToStableHlo : public ConversionPattern {
  ONNXReshapeOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXReshapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXReshapeOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Location loc = op->getLoc();
    Value data = operandAdaptor.getData();
    Type outputType = *op->result_type_begin();

    IndexExprBuilderForStableHlo createIE(rewriter, loc);
    ONNXReshapeOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr outputDims = shapeHelper.getOutputDims();
    SmallVector<Value> dims;
    IndexExpr::getValues(outputDims, dims);

    Type outputShapeType =
        RankedTensorType::get({(int64_t)dims.size()}, rewriter.getIndexType());
    Value shape = rewriter.create<shape::FromExtentsOp>(loc, dims);
    shape =
        rewriter.create<shape::ToExtentTensorOp>(loc, outputShapeType, shape);
    Value result = rewriter.create<stablehlo::DynamicReshapeOp>(
        loc, outputType, data, shape);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXReshapeOpToStableHloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpLoweringToStableHlo>(ctx);
}

} // namespace onnx_mlir
