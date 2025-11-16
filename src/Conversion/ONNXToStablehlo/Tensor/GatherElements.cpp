/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- GatherElements.cpp - Lowering GatherElements Op -------------===//
//
// Copyright 2023-2024
//
// =============================================================================
//
// This file lowers the ONNX GatherElements Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXGatherElementsOpLoweringToStablehlo : public ConversionPattern {
  ONNXGatherElementsOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXGatherElementsOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXGatherElementsOpAdaptor operandAdaptor(operands);
    ONNXGatherElementsOp gatherOp = cast<ONNXGatherElementsOp>(op);
    Location loc = op->getLoc();

    IndexExprBuilderForStablehlo createIE(rewriter, loc);
    ONNXGatherElementsOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");

    // Operands and attributes.
    Value data = operandAdaptor.getData();
    Value indices = operandAdaptor.getIndices();
    int64_t axisLit = gatherOp.getAxis();

    ShapedType inputType = mlir::cast<ShapedType>(data.getType());
    int64_t rank = inputType.getRank(); // indices has the same rank
    ShapedType indicesType = mlir::cast<ShapedType>(indices.getType());
    Type indexElemType = indicesType.getElementType();
    // Negative value means counting dimensions from the back.
    axisLit = axisLit < 0 ? axisLit + rank : axisLit;

    // make sure all index values >= 0
    Value zero = getShapedZero(loc, rewriter, indices);
    Value inputShape = shape::ShapeOfOp::create(rewriter, loc, data);
    Value indicesShape = shape::ShapeOfOp::create(rewriter, loc, indices);
    Value broadcastedAxisDimSize, axisDimSize;
    if (inputType.hasStaticShape()) {
      axisDimSize = stablehlo::ConstantOp::create(rewriter, loc,
          rewriter.getIntegerAttr(
              indexElemType, inputType.getDimSize(axisLit)));
    } else {
      axisDimSize =
          shape::GetExtentOp::create(rewriter, loc, inputShape, axisLit);
      axisDimSize =
          shape::GetExtentOp::create(rewriter, loc, inputShape, axisLit);
      axisDimSize =
          arith::IndexCastOp::create(rewriter, loc, indexElemType, axisDimSize);
      axisDimSize = tensor::FromElementsOp::create(rewriter, loc, axisDimSize);
      axisDimSize = stablehlo::ReshapeOp::create(rewriter, loc,
          RankedTensorType::get(SmallVector<int64_t>{}, indexElemType),
          axisDimSize);
    }
    if (indicesType.hasStaticShape()) {
      broadcastedAxisDimSize = stablehlo::BroadcastInDimOp::create(rewriter,
          loc, indicesType, axisDimSize, rewriter.getDenseI64ArrayAttr({}));
    } else {
      broadcastedAxisDimSize =
          stablehlo::DynamicBroadcastInDimOp::create(rewriter, loc, indicesType,
              axisDimSize, indicesShape, rewriter.getDenseI64ArrayAttr({}));
    }
    Value isNegative = stablehlo::CompareOp::create(
        rewriter, loc, indices, zero, stablehlo::ComparisonDirection::LT);
    Value positiveIndices = stablehlo::AddOp::create(
        rewriter, loc, indicesType, indices, broadcastedAxisDimSize);
    indices = stablehlo::SelectOp::create(
        rewriter, loc, indicesType, isNegative, positiveIndices, indices);

    // start indices
    Value toConcatIndexShape;
    SmallVector<Value> toConcatIndexShapeValueVec;
    for (int64_t i = 0; i < rank; i++) {
      toConcatIndexShapeValueVec.push_back(
          shape::GetExtentOp::create(rewriter, loc, indicesShape, i));
    }
    toConcatIndexShapeValueVec.push_back(
        arith::ConstantIndexOp::create(rewriter, loc, 1));
    toConcatIndexShape = tensor::FromElementsOp::create(
        rewriter, loc, toConcatIndexShapeValueVec);

    ArrayRef<int64_t> indicesShapeVec = indicesType.getShape();
    SmallVector<int64_t> toConcatIndexShapeVec(
        indicesShapeVec.begin(), indicesShapeVec.end());
    toConcatIndexShapeVec.push_back(1);
    RankedTensorType toConcatIndexType =
        RankedTensorType::get(toConcatIndexShapeVec, indexElemType);

    SmallVector<Value> toConcat;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      if (i == axisLit) {
        toConcat.push_back(stablehlo::DynamicReshapeOp::create(
            rewriter, loc, toConcatIndexType, indices, toConcatIndexShape));
      } else {
        toConcat.push_back(
            stablehlo::DynamicIotaOp::create(rewriter, loc, toConcatIndexType,
                toConcatIndexShape, rewriter.getI64IntegerAttr(i)));
      }
    }
    auto gatherIndicies = stablehlo::ConcatenateOp::create(
        rewriter, loc, toConcat, static_cast<uint64_t>(inputType.getRank()));

    // dimsAttr
    SmallVector<int64_t> collapsedDims;
    SmallVector<int64_t> startIndexMap;
    for (int64_t i = 0; i < rank; i++) {
      collapsedDims.push_back(i);
      startIndexMap.push_back(i);
    }
    auto dimsAttr =
        stablehlo::GatherDimensionNumbersAttr::get(rewriter.getContext(),
            /*offsetDims=*/{},
            /*collapsedSliceDims=*/collapsedDims,
            /*operandBatchingDims=*/{},
            /*startIndicesBatchingDims=*/{},
            /*startIndexMap=*/startIndexMap,
            /*indexVecDim=*/rank);
    SmallVector<int64_t> sliceSizes(inputType.getRank(), 1);

    Value gatherValue = stablehlo::GatherOp::create(rewriter, loc, outputType,
        data, gatherIndicies, dimsAttr,
        rewriter.getDenseI64ArrayAttr(sliceSizes));
    rewriter.replaceOp(op, gatherValue);
    return success();
  }
};

} // namespace

void populateLoweringONNXGatherElementsOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXGatherElementsOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
