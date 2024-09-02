/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Gather.cpp - Lowering Gather Op ---------------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Gather Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXGatherOp is mainly implemented using Stablehlo TorchIndexSelectOp
struct ONNXGatherOpLoweringToStablehlo : public ConversionPattern {
  ONNXGatherOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXGatherOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXGatherOpAdaptor operandAdaptor(operands);
    ONNXGatherOp gatherOp = cast<ONNXGatherOp>(op);
    Location loc = op->getLoc();

    // Is it unused?
    IndexExprBuilderForStablehlo createIE(rewriter, loc);
    ONNXGatherOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Type outputType = *op->result_type_begin();
    if (!isRankedShapedType(outputType))
      return rewriter.notifyMatchFailure(op, "Expected Ranked ShapedType");

    // Operands and attributes.
    Value data = operandAdaptor.getData();
    Value indices = operandAdaptor.getIndices();
    int64_t axisLit = gatherOp.getAxis();

    ShapedType inputType = mlir::cast<ShapedType>(data.getType());
    int64_t dataRank = inputType.getRank();
    ShapedType indicesType = mlir::cast<ShapedType>(indices.getType());
    // Negative value means counting dimensions from the back.
    axisLit = axisLit < 0 ? axisLit + dataRank : axisLit;

    // start indices
    Value zero = getShapedZero(loc, rewriter, indices);
    Value axisDimSize;
    if (!inputType.isDynamicDim(axisLit)) {
      int64_t axisDimSizeLit = inputType.getShape()[axisLit];
      axisDimSize = getShapedInt(loc, rewriter, axisDimSizeLit, indices);
    } else {
      Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, data);
      Value indicesShape = rewriter.create<shape::ShapeOfOp>(loc, indices);
      Value axisDimSizeIndexValue =
          rewriter.create<shape::GetExtentOp>(loc, inputShape, axisLit);
      Value axisDimSizeValue = rewriter.create<arith::IndexCastOp>(
          loc, indicesType.getElementType(), axisDimSizeIndexValue);
      axisDimSizeValue = rewriter.create<tensor::FromElementsOp>(loc,
          RankedTensorType::get({}, indicesType.getElementType()),
          axisDimSizeValue);
      axisDimSize = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(loc,
          indicesType, axisDimSizeValue, indicesShape,
          rewriter.getDenseI64ArrayAttr({}));
    }
    Value greaterOp = rewriter.create<stablehlo::CompareOp>(
        loc, indices, zero, stablehlo::ComparisonDirection::LT);
    Value positiveIndices = rewriter.create<stablehlo::AddOp>(
        loc, indicesType, indices, axisDimSize);
    Value startIndices = rewriter.create<stablehlo::SelectOp>(
        loc, indicesType, greaterOp, positiveIndices, indices);

    Value gatherValue = rewriter.create<stablehlo::TorchIndexSelectOp>(loc,
        outputType, data, startIndices, rewriter.getI64IntegerAttr(axisLit),
        rewriter.getI64IntegerAttr(0));
    rewriter.replaceOp(op, gatherValue);
    return success();
  }
};

} // namespace

void populateLoweringONNXGatherOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXGatherOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
