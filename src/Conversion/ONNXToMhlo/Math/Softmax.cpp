/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Softmax.cpp - Softmax Op ---------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Broadcasts value tensor 'value' to the shape of 'resultType'. If
// 'shapeValue' is initialized, creates a dynamic broadcast, otherwise creates
// a static broadcast.
Value broadcastToOriginShape(Location loc, RankedTensorType resultType,
    Value value, Value shapeValue, SmallVector<int64_t> &reducedShape,
    PatternRewriter &rewriter) { // NOLINT
  auto dims = rewriter.getI64TensorAttr(reducedShape);
  if (shapeValue) {
    return rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(
        loc, resultType, value, shapeValue, dims);
  }
  assert(resultType.hasStaticShape());
  return rewriter.create<mhlo::BroadcastInDimOp>(loc, resultType, value, dims);
}

// Create "mhlo.reduce", "operand" is reduce input and "zero" is init value,
// reduce from operand to operand[reduceIdx].
template <typename ReduceOp>
Value createReduce(Location loc, Value operand, Value zero,
    SmallVector<int64_t> &reduceShape, int64_t reduceIdx,
    PatternRewriter &rewriter) {
  auto operandType = operand.getType().cast<RankedTensorType>();
  Type reduceResultType =
      RankedTensorType::get(reduceShape, operandType.getElementType());
  mhlo::ReduceOp reduce = rewriter.create<mhlo::ReduceOp>(loc, reduceResultType,
      operand, zero, rewriter.getI64TensorAttr({reduceIdx}));

  // setup "mhlo.reduce"'s body
  Region &region = reduce.body();
  Block &block = region.emplaceBlock();
  RankedTensorType blockArgumentType =
      RankedTensorType::get({}, operandType.getElementType());
  block.addArgument(blockArgumentType, loc);
  block.addArgument(blockArgumentType, loc);
  auto firstArgument = *block.args_begin();
  auto secondArgument = *block.args_rbegin();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value reduceResult =
        rewriter.create<ReduceOp>(loc, firstArgument, secondArgument);
    rewriter.create<mhlo::ReturnOp>(loc, reduceResult);
  }
  return reduce.getResult(0);
}

static void emitInstForSoftmaxV13(ConversionPatternRewriter &rewriter,
    Operation *op, Location loc, Value input, Value zero, Value negInfinity,
    int64_t axis) {
  auto operandType = input.getType().dyn_cast<RankedTensorType>();
  int64_t rank = operandType.getRank();

  SmallVector<int64_t> dimensionsWithoutFeature;
  for (int64_t i = 0; i < rank; i++) {
    if (i != axis) {
      dimensionsWithoutFeature.push_back(i);
    }
  }
  SmallVector<int64_t> reduceShape;
  for (int64_t i = 0; i < rank - 1; i++) {
    reduceShape.push_back(operandType.getDimSize(dimensionsWithoutFeature[i]));
  }

  // max[X]
  Value maxValue = createReduce<mhlo::MaxOp>(
      loc, input, negInfinity, reduceShape, axis, rewriter);
  Value shapeValue;
  if (!operandType.hasStaticShape()) {
    shapeValue = rewriter.create<shape::ShapeOfOp>(loc, input);
  }
  auto maxValueBroadcast = broadcastToOriginShape(loc, operandType, maxValue,
      shapeValue, dimensionsWithoutFeature, rewriter);
  // X - max[X]
  auto shiftedLogits =
      rewriter.create<mhlo::SubOp>(loc, input, maxValueBroadcast);
  // exp(X - max[X])
  auto expLogits = rewriter.create<mhlo::ExpOp>(loc, shiftedLogits);
  // sum[exp(X - max[X])]
  auto expSum = createReduce<mhlo::AddOp>(
      loc, expLogits, zero, reduceShape, axis, rewriter);
  auto expSumBroadcast = broadcastToOriginShape(
      loc, operandType, expSum, shapeValue, dimensionsWithoutFeature, rewriter);
  // exp(X - max[X]) / sum[exp(X - max[X])]
  auto divOp = rewriter.create<mhlo::DivOp>(loc, expLogits, expSumBroadcast);
  rewriter.replaceOp(op, divOp->getResults());
}

struct ONNXSoftmaxOpLoweringToMhlo : public ConversionPattern {
  ONNXSoftmaxOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSoftmaxOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // softmax(x) = let max_x = max(x) in
    //                let exp_x = exp(x - max_x) in
    //                  let sum = sum(exp_x) in
    //                    exp_x / sum
    ONNXSoftmaxOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Value inp = operandAdaptor.input();
    auto inputType = inp.getType().cast<RankedTensorType>();
    if (inputType == nullptr) {
      return failure();
    }
    auto rank = inputType.getRank();
    int64_t axis = operandAdaptor.axis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);

    // Get opset number. Default is opset 11.
    int64_t opset = 11;
    IntegerAttr opsetAttr = op->getAttrOfType<::mlir::Attribute>("onnx_opset")
                                .dyn_cast_or_null<IntegerAttr>();
    if (opsetAttr)
      opset = opsetAttr.getValue().getSExtValue();

    auto loc = op->getLoc();
    auto elementType = inputType.getElementType().cast<FloatType>();
    Value zero =
        rewriter.create<mhlo::ConstOp>(loc, rewriter.getZeroAttr(elementType));
    Value negInfinity = rewriter.create<mhlo::ConstOp>(
        loc, rewriter.getFloatAttr(
                 elementType, APFloat::getInf(elementType.getFloatSemantics(),
                                  /*isNegative=*/true)));
    if (opset < 13) {
      op->emitError() << "Not Support SoftmaxBeforeV13\n";
      return failure();
    }
    emitInstForSoftmaxV13(rewriter, op, loc, inp, zero, negInfinity, axis);

    return success();
  }
};

} // namespace

void populateLoweringONNXSoftmaxOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSoftmaxOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
