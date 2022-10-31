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
    ONNXSliceOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    ONNXSliceOp sliceOp = llvm::cast<ONNXSliceOp>(op);
    Location loc = op->getLoc();

    // shape helper
    IndexExprScope scope(&rewriter, loc);
    ONNXSliceOpShapeHelper shapeHelper(&sliceOp, &rewriter,
        getDenseElementAttributeFromMhloValue,
        loadValuefromArrayAtIndexWithMhlo, &scope);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(!failed(shapecomputed) && "shape helper failed");

    Value data = sliceOp.data();

    assert(isRankedShapedType(data.getType()) &&
           "data must be ranked Shaped Type");
    ShapedType dataType = data.getType().cast<ShapedType>();
    int64_t rank = dataType.getRank();
    Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, data);
    Value zero = rewriter.create<mhlo::ConstantOp>(
        loc, DenseIntElementsAttr::get(
                 RankedTensorType::get({1}, rewriter.getI64Type()),
                 ArrayRef<int64_t>{0}));
    SmallVector<IndexExpr, 4> startIEs;
    SmallVector<IndexExpr, 4> endIEs;
    SmallVector<IndexExpr, 4> stepIEs;

    MemRefBoundsIndexCapture dataBounds(data);
    for (int64_t i = 0; i < rank; ++i) {
      DimIndexExpr dimInput(dataBounds.getDim(i));
      IndexExpr start = shapeHelper.starts[i];
      IndexExpr end = shapeHelper.ends[i];
      IndexExpr step = shapeHelper.steps[i];

      IndexExpr isNonNegativeIE = step >= 0;
      IndexExpr startFinal =
          IndexExpr::select(isNonNegativeIE, start, dimInput - 1 - start);
      IndexExpr endFinal =
          IndexExpr::select(isNonNegativeIE, end, dimInput - 1 - end);
      IndexExpr stepFinal =
          IndexExpr::select(isNonNegativeIE, step, LiteralIndexExpr(0) - step);
      startIEs.push_back(startFinal);
      endIEs.push_back(endFinal);
      stepIEs.push_back(stepFinal);

      Value reversedData = rewriter.create<mhlo::ReverseOp>(loc, data,
          DenseIntElementsAttr::get(
              RankedTensorType::get({1}, rewriter.getI64Type()),
              ArrayRef<int64_t>{i}));
      if (step.isLiteral()) {
        if (step.getLiteral() < 0)
          data = reversedData;
      } else {
        Value stepValue = rewriter.create<shape::FromExtentsOp>(
            loc, ArrayRef<Value>{step.getValue()});
        stepValue = rewriter.create<shape::ToExtentTensorOp>(loc,
            RankedTensorType::get({1}, rewriter.getIndexType()), stepValue);
        stepValue = rewriter.create<arith::IndexCastOp>(
            loc, RankedTensorType::get({1}, rewriter.getI64Type()), stepValue);
        Value isNonNegative = rewriter.create<mhlo::CompareOp>(
            loc, stepValue, zero, mhlo::ComparisonDirection::GE);
        Value broadcastedIsNonNegative =
            rewriter.create<mhlo::DynamicBroadcastInDimOp>(loc,
                RankedTensorType::get(
                    dataType.getShape(), rewriter.getI1Type()),
                isNonNegative, inputShape, rewriter.getI64TensorAttr({0}));
        data = rewriter.create<mhlo::SelectOp>(
            loc, broadcastedIsNonNegative, data, reversedData);
      }
    }

    auto createIndicesValue = [&](SmallVector<IndexExpr, 4> &IEs,
                                  Value &indicesValue) {
      Type indexTensorType =
          RankedTensorType::get({rank}, rewriter.getIndexType());
      Type I64TensorType = RankedTensorType::get({rank}, rewriter.getI64Type());
      if (IndexExpr::isLiteral(IEs)) {
        SmallVector<int64_t, 4> values;
        for (int i = 0; i < rank; i++) {
          values.push_back(IEs[i].getLiteral());
        }
        indicesValue = rewriter.create<mhlo::ConstantOp>(
            loc, DenseIntElementsAttr::get(I64TensorType, values));
      } else {
        SmallVector<Value, 4> values;
        IndexExpr::getValues(IEs, values);
        indicesValue = rewriter.create<shape::FromExtentsOp>(loc, values);
        indicesValue = rewriter.create<shape::ToExtentTensorOp>(
            loc, indexTensorType, indicesValue);
        indicesValue = rewriter.create<arith::IndexCastOp>(
            loc, I64TensorType, indicesValue);
      }
    };
    Value startIndices, endIndices, stepIndices;
    createIndicesValue(startIEs, startIndices);
    createIndicesValue(endIEs, endIndices);
    createIndicesValue(stepIEs, stepIndices);

    Type outputType = *op->result_type_begin();
    Value sliceValue = rewriter.create<mhlo::RealDynamicSliceOp>(
        loc, outputType, data, startIndices, endIndices, stepIndices);
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
