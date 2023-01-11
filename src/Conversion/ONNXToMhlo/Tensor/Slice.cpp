/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Slice.cpp - Lowering Slice Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Slice Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXSliceOp(A) is mainly implemented using MHLO sliceOp, and follows the
// onnx definition of slice.
struct ONNXSliceOpLoweringToMhlo : public ConversionPattern {
  ONNXSliceOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSliceOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSliceOpAdaptor operandAdaptor(operands);
    ONNXSliceOp sliceOp = llvm::cast<ONNXSliceOp>(op);
    Location loc = op->getLoc();

    Value data = sliceOp.data();
    Value starts = sliceOp.starts();
    Value axes = sliceOp.axes();
    Value ends = sliceOp.ends();
    Value steps = sliceOp.steps();

    assert(isRankedShapedType(data.getType()) &&
           "data must be ranked Shaped Type");
    ShapedType dataType = data.getType().cast<ShapedType>();
    int64_t rank = dataType.getRank();
    Type indiceElementType = rewriter.getI64Type();
    Value zero = rewriter.create<mhlo::ConstantOp>(loc,
        DenseIntElementsAttr::get(RankedTensorType::get({1}, indiceElementType),
            ArrayRef<int64_t>{0}));
    Value one = rewriter.create<mhlo::ConstantOp>(loc,
        DenseIntElementsAttr::get(RankedTensorType::get({1}, indiceElementType),
            ArrayRef<int64_t>{1}));
    SmallVector<Value, 4> stepValues;
    SmallVector<Value, 4> beginValues;
    SmallVector<Value, 4> endValues;
    SmallVector<int64_t, 2> axesIntLitToIdx(rank, -1);
    SmallVector<Value, 4> indices;

    if (axes.getType().isa<NoneType>()) {
      // If `axes` are omitted, they are set to `[0, ..., nDim-1]`."
      for (int64_t i = 0; i < rank; ++i)
        axesIntLitToIdx[i] = i;
    } else if (auto valueAttribute = getElementAttributeFromONNXValue(axes)) {
      // If `axes` are constants, read them."
      int64_t idx = 0;
      for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
        int64_t axis = value.cast<IntegerAttr>().getInt();
        if (axis < 0)
          axis += rank;
        assert((axis >= 0 && axis < (int64_t)rank) &&
               "Axes contains an out-of-bound index");
        axesIntLitToIdx[axis] = idx++;
      }
    } else {
      assert(false && "Axes must be known at compile time");
    }

    Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, data);

    for (int64_t i = 0; i < rank; ++i) {
      Value dimValue;
      if (dataType.getShape()[i] != ShapedType::kDynamicSize)
        dimValue = rewriter.create<mhlo::ConstantOp>(
            loc, DenseIntElementsAttr::get(
                     RankedTensorType::get({1}, indiceElementType),
                     ArrayRef<int64_t>{dataType.getShape()[i]}));
      else {
        Value dimIndexValue =
            rewriter.create<shape::GetExtentOp>(loc, inputShape, i);
        dimValue = rewriter.create<shape::FromExtentsOp>(loc, dimIndexValue);
        dimValue = rewriter.create<shape::ToExtentTensorOp>(
            loc, RankedTensorType::get({1}, rewriter.getIndexType()), dimValue);
        dimValue = rewriter.create<arith::IndexCastOp>(
            loc, RankedTensorType::get({1}, indiceElementType), dimValue);
      }
      if (axesIntLitToIdx[i] == -1) {
        beginValues.push_back(zero);
        stepValues.push_back(one);
        endValues.push_back(dimValue);
      } else {
        Value beginValue = rewriter.create<mhlo::SliceOp>(loc,
            RankedTensorType::get({1}, indiceElementType), starts,
            DenseIntElementsAttr::get(
                RankedTensorType::get({1}, indiceElementType),
                ArrayRef<int64_t>{axesIntLitToIdx[i]}),
            DenseIntElementsAttr::get(
                RankedTensorType::get({1}, indiceElementType),
                ArrayRef<int64_t>{axesIntLitToIdx[i] + 1}),
            DenseIntElementsAttr::get(
                RankedTensorType::get({1}, indiceElementType),
                ArrayRef<int64_t>{1}));
        Value stepValue = rewriter.create<mhlo::SliceOp>(loc,
            RankedTensorType::get({1}, indiceElementType), steps,
            DenseIntElementsAttr::get(
                RankedTensorType::get({1}, indiceElementType),
                ArrayRef<int64_t>{axesIntLitToIdx[i]}),
            DenseIntElementsAttr::get(
                RankedTensorType::get({1}, indiceElementType),
                ArrayRef<int64_t>{axesIntLitToIdx[i] + 1}),
            DenseIntElementsAttr::get(
                RankedTensorType::get({1}, indiceElementType),
                ArrayRef<int64_t>{1}));
        Value endValue = rewriter.create<mhlo::SliceOp>(loc,
            RankedTensorType::get({1}, indiceElementType), ends,
            DenseIntElementsAttr::get(
                RankedTensorType::get({1}, indiceElementType),
                ArrayRef<int64_t>{axesIntLitToIdx[i]}),
            DenseIntElementsAttr::get(
                RankedTensorType::get({1}, indiceElementType),
                ArrayRef<int64_t>{axesIntLitToIdx[i] + 1}),
            DenseIntElementsAttr::get(
                RankedTensorType::get({1}, indiceElementType),
                ArrayRef<int64_t>{1}));
        Value isNegativeStepValue = rewriter.create<mhlo::CompareOp>(
            loc, stepValue, zero, mhlo::ComparisonDirection::LT);
        Value broadcastedIsNegativeValue =
            rewriter.create<mhlo::DynamicBroadcastInDimOp>(loc,
                RankedTensorType::get(
                    dataType.getShape(), rewriter.getI1Type()),
                isNegativeStepValue, inputShape,
                rewriter.getI64TensorAttr({0}));
        Value negatedStepValue = rewriter.create<mhlo::NegOp>(loc, stepValue);
        Value negatedStartValue =
            rewriter.create<mhlo::AddOp>(loc, endValue, one);
        Value negatedEndValue =
            rewriter.create<mhlo::AddOp>(loc, beginValue, one);
        Value reversedData = rewriter.create<mhlo::ReverseOp>(loc, data,
            DenseIntElementsAttr::get(
                RankedTensorType::get({1}, indiceElementType),
                ArrayRef<int64_t>{i}));
        beginValue = rewriter.create<mhlo::SelectOp>(
            loc, isNegativeStepValue, negatedStartValue, beginValue);
        endValue = rewriter.create<mhlo::SelectOp>(
            loc, isNegativeStepValue, negatedEndValue, endValue);
        stepValue = rewriter.create<mhlo::SelectOp>(
            loc, isNegativeStepValue, negatedStepValue, stepValue);
        data = rewriter.create<mhlo::SelectOp>(
            loc, broadcastedIsNegativeValue, reversedData, data);
        Value exceedDimValue = rewriter.create<mhlo::CompareOp>(
            loc, endValue, dimValue, mhlo::ComparisonDirection::GT);
        Value clampedEndValue = rewriter.create<mhlo::SelectOp>(
            loc, exceedDimValue, dimValue, endValue);
        Value negDimValue = rewriter.create<mhlo::CompareOp>(
            loc, clampedEndValue, zero, mhlo::ComparisonDirection::LT);
        Value posDimValue =
            rewriter.create<mhlo::AddOp>(loc, clampedEndValue, dimValue);
        Value resultEndValue = rewriter.create<mhlo::SelectOp>(
            loc, negDimValue, posDimValue, clampedEndValue);
        Value negStartValue = rewriter.create<mhlo::CompareOp>(
            loc, beginValue, zero, mhlo::ComparisonDirection::LT);
        Value posStartValue =
            rewriter.create<mhlo::AddOp>(loc, beginValue, dimValue);
        Value resultBeginValue = rewriter.create<mhlo::SelectOp>(
            loc, negStartValue, posStartValue, beginValue);
        beginValues.push_back(resultBeginValue);
        stepValues.push_back(stepValue);
        endValues.push_back(resultEndValue);
      }
    }

    Type outputType = *op->result_type_begin();
    auto start_indices = rewriter.create<mhlo::ConcatenateOp>(loc,
        RankedTensorType::get(
            {static_cast<int64_t>(beginValues.size())}, indiceElementType),
        beginValues, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    auto end_indices = rewriter.create<mhlo::ConcatenateOp>(loc,
        RankedTensorType::get(
            {static_cast<int64_t>(endValues.size())}, indiceElementType),
        endValues, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    auto step_indices = rewriter.create<mhlo::ConcatenateOp>(loc,
        RankedTensorType::get(
            {static_cast<int64_t>(stepValues.size())}, indiceElementType),
        stepValues, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    Value sliceValue = rewriter.create<mhlo::RealDynamicSliceOp>(
        loc, outputType, data, start_indices, end_indices, step_indices);
    rewriter.replaceOp(op, sliceValue);
    return success();
  }
};

} // namespace

void populateLoweringONNXSliceOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSliceOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
