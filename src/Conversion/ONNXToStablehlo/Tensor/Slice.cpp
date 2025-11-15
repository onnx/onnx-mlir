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
    Value zero = stablehlo::ConstantOp::create(rewriter, loc,
        DenseElementsAttr::get(RankedTensorType::get({1}, indexElementType),
            ArrayRef<int64_t>{0}));
    Value one = stablehlo::ConstantOp::create(rewriter, loc,
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

    Value inputShape = shape::ShapeOfOp::create(rewriter, loc, data);

    for (int64_t i = 0; i < rank; ++i) {
      Value dimValue;
      if (dataType.getShape()[i] != ShapedType::kDynamic)
        dimValue = stablehlo::ConstantOp::create(rewriter, loc,
            DenseElementsAttr::get(RankedTensorType::get({1}, indexElementType),
                ArrayRef<int64_t>{dataType.getShape()[i]}));
      else {
        Value dimIndexValue =
            shape::GetExtentOp::create(rewriter, loc, inputShape, i);
        dimValue = shape::FromExtentsOp::create(rewriter, loc, dimIndexValue);
        dimValue = shape::ToExtentTensorOp::create(rewriter, loc,
            RankedTensorType::get({1}, rewriter.getIndexType()), dimValue);
        dimValue = arith::IndexCastOp::create(rewriter, loc,
            RankedTensorType::get({1}, indexElementType), dimValue);
      }
      if (axesIntLitToIdx[i] == -1) {
        beginValues.push_back(zero);
        stepValues.push_back(one);
        endValues.push_back(dimValue);
      } else {
        Value beginValue = stablehlo::SliceOp::create(rewriter, loc,
            RankedTensorType::get({1}, indexElementType), starts,
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i]}),
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i] + 1}),
            DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1}));
        Value stepValue = stablehlo::SliceOp::create(rewriter, loc,
            RankedTensorType::get({1}, indexElementType), steps,
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i]}),
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i] + 1}),
            DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1}));
        Value endValue = stablehlo::SliceOp::create(rewriter, loc,
            RankedTensorType::get({1}, indexElementType), ends,
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i]}),
            DenseI64ArrayAttr::get(
                context, ArrayRef<int64_t>{axesIntLitToIdx[i] + 1}),
            DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1}));
        Value isNegativeStepValue = stablehlo::CompareOp::create(
            rewriter, loc, stepValue, zero, stablehlo::ComparisonDirection::LT);
        Value broadcastedIsNegativeValue =
            stablehlo::DynamicBroadcastInDimOp::create(rewriter, loc,
                RankedTensorType::get(
                    dataType.getShape(), rewriter.getI1Type()),
                isNegativeStepValue, inputShape,
                rewriter.getDenseI64ArrayAttr({0}));
        Value negatedStepValue =
            stablehlo::NegOp::create(rewriter, loc, stepValue);
        Value negatedStartValue =
            stablehlo::AddOp::create(rewriter, loc, endValue, one);
        Value negatedEndValue =
            stablehlo::AddOp::create(rewriter, loc, beginValue, one);
        Value reversedData = stablehlo::ReverseOp::create(rewriter, loc, data,
            DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{i}));
        beginValue = stablehlo::SelectOp::create(
            rewriter, loc, isNegativeStepValue, negatedStartValue, beginValue);
        endValue = stablehlo::SelectOp::create(
            rewriter, loc, isNegativeStepValue, negatedEndValue, endValue);
        stepValue = stablehlo::SelectOp::create(
            rewriter, loc, isNegativeStepValue, negatedStepValue, stepValue);
        data = stablehlo::SelectOp::create(
            rewriter, loc, broadcastedIsNegativeValue, reversedData, data);
        Value exceedDimValue = stablehlo::CompareOp::create(rewriter, loc,
            endValue, dimValue, stablehlo::ComparisonDirection::GT);
        Value clampedEndValue = stablehlo::SelectOp::create(
            rewriter, loc, exceedDimValue, dimValue, endValue);
        Value negDimValue = stablehlo::CompareOp::create(rewriter, loc,
            clampedEndValue, zero, stablehlo::ComparisonDirection::LT);
        Value posDimValue =
            stablehlo::AddOp::create(rewriter, loc, clampedEndValue, dimValue);
        Value resultEndValue = stablehlo::SelectOp::create(
            rewriter, loc, negDimValue, posDimValue, clampedEndValue);
        Value negStartValue = stablehlo::CompareOp::create(rewriter, loc,
            beginValue, zero, stablehlo::ComparisonDirection::LT);
        Value posStartValue =
            stablehlo::AddOp::create(rewriter, loc, beginValue, dimValue);
        Value resultBeginValue = stablehlo::SelectOp::create(
            rewriter, loc, negStartValue, posStartValue, beginValue);
        beginValues.push_back(resultBeginValue);
        stepValues.push_back(stepValue);
        endValues.push_back(resultEndValue);
      }
    }

    Type outputType = *op->result_type_begin();
    auto start_indices = stablehlo::ConcatenateOp::create(rewriter, loc,
        RankedTensorType::get(
            {static_cast<int64_t>(beginValues.size())}, indexElementType),
        beginValues, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    auto end_indices = stablehlo::ConcatenateOp::create(rewriter, loc,
        RankedTensorType::get(
            {static_cast<int64_t>(endValues.size())}, indexElementType),
        endValues, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    auto step_indices = stablehlo::ConcatenateOp::create(rewriter, loc,
        RankedTensorType::get(
            {static_cast<int64_t>(stepValues.size())}, indexElementType),
        stepValues, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    Value sliceValue = stablehlo::RealDynamicSliceOp::create(rewriter, loc,
        outputType, data, start_indices, end_indices, step_indices);
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
