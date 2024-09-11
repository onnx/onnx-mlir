/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Slice.cpp - Lowering Slice Op----------------------=== //
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Slice Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXSliceOp(A) is mainly implemented using Stablehlo sliceOp, and follows the
// onnx definition of slice.
struct ONNXSliceOpLoweringToStablehlo : public ConversionPattern {
  ONNXSliceOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSliceOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSliceOpAdaptor operandAdaptor(operands);
    ONNXSliceOp sliceOp = llvm::cast<ONNXSliceOp>(op);
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();

    Value data = sliceOp.getData();
    Value starts = sliceOp.getStarts();
    Value axes = sliceOp.getAxes();
    Value ends = sliceOp.getEnds();
    Value steps = sliceOp.getSteps();

    assert(isRankedShapedType(data.getType()) &&
           "data must be ranked Shaped Type");
    ShapedType dataType = mlir::cast<ShapedType>(data.getType());
    int64_t rank = dataType.getRank();
    Type indexElementType = rewriter.getI64Type();
    Value zero = rewriter.create<stablehlo::ConstantOp>(loc,
        DenseElementsAttr::get(RankedTensorType::get({1}, indexElementType),
            ArrayRef<int64_t>{0}));
    Value one = rewriter.create<stablehlo::ConstantOp>(loc,
        DenseElementsAttr::get(RankedTensorType::get({1}, indexElementType),
            ArrayRef<int64_t>{1}));
    SmallVector<Value, 4> stepValues;
    SmallVector<Value, 4> beginValues;
    SmallVector<Value, 4> endValues;
    SmallVector<int64_t, 2> axesIntLitToIdx(rank, -1);
    SmallVector<Value, 4> indices;

    if (mlir::isa<NoneType>(axes.getType())) {
      // If `axes` are omitted, they are set to `[0, ..., nDim-1]`."
      for (int64_t i = 0; i < rank; ++i)
        axesIntLitToIdx[i] = i;
    } else if (auto valueAttribute = getElementAttributeFromONNXValue(axes)) {
      // If `axes` are constants, read them."
      int64_t idx = 0;
      for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
        int64_t axis = mlir::cast<IntegerAttr>(value).getInt();
        if (axis < 0)
          axis += rank;
        assert((axis >= 0 && axis < static_cast<int64_t>(rank)) &&
               "Axes contains an out-of-bound index");
        axesIntLitToIdx[axis] = idx++;
      }
    } else {
      assert(false && "Axes must be known at compile time");
    }

    Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, data);

    for (int64_t i = 0; i < rank; ++i) {
      Value dimValue;
      if (dataType.getShape()[i] != ShapedType::kDynamic)
        dimValue = rewriter.create<stablehlo::ConstantOp>(loc,
            DenseElementsAttr::get(RankedTensorType::get({1}, indexElementType),
                ArrayRef<int64_t>{dataType.getShape()[i]}));
      else {
        Value dimIndexValue =
            rewriter.create<shape::GetExtentOp>(loc, inputShape, i);
        dimValue = rewriter.create<shape::FromExtentsOp>(loc, dimIndexValue);
        dimValue = rewriter.create<shape::ToExtentTensorOp>(
            loc, RankedTensorType::get({1}, rewriter.getIndexType()), dimValue);
        dimValue = rewriter.create<arith::IndexCastOp>(
            loc, RankedTensorType::get({1}, indexElementType), dimValue);
      }
      if (axesIntLitToIdx[i] == -1) {
        beginValues.push_back(zero);
        stepValues.push_back(one);
        endValues.push_back(dimValue);
      } else {
        Value beginValue = rewriter.create<stablehlo::SliceOp>(loc,
            RankedTensorType::get({1}, indexElementType), starts,
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i]}),
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i] + 1}),
            DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1}));
        Value stepValue = rewriter.create<stablehlo::SliceOp>(loc,
            RankedTensorType::get({1}, indexElementType), steps,
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i]}),
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i] + 1}),
            DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1}));
        Value endValue = rewriter.create<stablehlo::SliceOp>(loc,
            RankedTensorType::get({1}, indexElementType), ends,
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i]}),
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i] + 1}),
            DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1}));
        Value isNegativeStepValue = rewriter.create<stablehlo::CompareOp>(
            loc, stepValue, zero, stablehlo::ComparisonDirection::LT);
        Value broadcastedIsNegativeValue =
            rewriter.create<stablehlo::DynamicBroadcastInDimOp>(loc,
                RankedTensorType::get(
                    dataType.getShape(), rewriter.getI1Type()),
                isNegativeStepValue, inputShape,
                rewriter.getDenseI64ArrayAttr({0}));
        Value negatedStepValue =
            rewriter.create<stablehlo::NegOp>(loc, stepValue);
        Value negatedStartValue =
            rewriter.create<stablehlo::AddOp>(loc, endValue, one);
        Value negatedEndValue =
            rewriter.create<stablehlo::AddOp>(loc, beginValue, one);
        Value reversedData = rewriter.create<stablehlo::ReverseOp>(
            loc, data, DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{i}));
        beginValue = rewriter.create<stablehlo::SelectOp>(
            loc, isNegativeStepValue, negatedStartValue, beginValue);
        endValue = rewriter.create<stablehlo::SelectOp>(
            loc, isNegativeStepValue, negatedEndValue, endValue);
        stepValue = rewriter.create<stablehlo::SelectOp>(
            loc, isNegativeStepValue, negatedStepValue, stepValue);
        data = rewriter.create<stablehlo::SelectOp>(
            loc, broadcastedIsNegativeValue, reversedData, data);
        Value exceedDimValue = rewriter.create<stablehlo::CompareOp>(
            loc, endValue, dimValue, stablehlo::ComparisonDirection::GT);
        Value clampedEndValue = rewriter.create<stablehlo::SelectOp>(
            loc, exceedDimValue, dimValue, endValue);
        Value negDimValue = rewriter.create<stablehlo::CompareOp>(
            loc, clampedEndValue, zero, stablehlo::ComparisonDirection::LT);
        Value posDimValue =
            rewriter.create<stablehlo::AddOp>(loc, clampedEndValue, dimValue);
        Value resultEndValue = rewriter.create<stablehlo::SelectOp>(
            loc, negDimValue, posDimValue, clampedEndValue);
        Value negStartValue = rewriter.create<stablehlo::CompareOp>(
            loc, beginValue, zero, stablehlo::ComparisonDirection::LT);
        Value posStartValue =
            rewriter.create<stablehlo::AddOp>(loc, beginValue, dimValue);
        Value resultBeginValue = rewriter.create<stablehlo::SelectOp>(
            loc, negStartValue, posStartValue, beginValue);
        beginValues.push_back(resultBeginValue);
        stepValues.push_back(stepValue);
        endValues.push_back(resultEndValue);
      }
    }

    Type outputType = *op->result_type_begin();
    auto start_indices = rewriter.create<stablehlo::ConcatenateOp>(loc,
        RankedTensorType::get(
            {static_cast<int64_t>(beginValues.size())}, indexElementType),
        beginValues, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    auto end_indices = rewriter.create<stablehlo::ConcatenateOp>(loc,
        RankedTensorType::get(
            {static_cast<int64_t>(endValues.size())}, indexElementType),
        endValues, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    auto step_indices = rewriter.create<stablehlo::ConcatenateOp>(loc,
        RankedTensorType::get(
            {static_cast<int64_t>(stepValues.size())}, indexElementType),
        stepValues, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    Value sliceValue = rewriter.create<stablehlo::RealDynamicSliceOp>(
        loc, outputType, data, start_indices, end_indices, step_indices);
    rewriter.replaceOp(op, sliceValue);
    return success();
  }
};

} // namespace

void populateLoweringONNXSliceOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSliceOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
