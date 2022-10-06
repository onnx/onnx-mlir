/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Tile.cpp - Lowering Tile Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Tile Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

// ONNXTileOp(A) is mainly implemented using MHLO broadcastOp & reshapeOp
struct ONNXTileOpLoweringToMhlo : public ConversionPattern {
  ONNXTileOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTileOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTileOpAdaptor operandAdaptor(operands);
    ONNXTileOp tileOp = cast<ONNXTileOp>(op);
    Location loc = op->getLoc();

    ONNXTileOpShapeHelper shapeHelper(&tileOp);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapecomputed;
    assert(!failed(shapecomputed) && "shapehelper failed");

    // Convert the output type to MemRefType.
    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");

    Value input = tileOp.input();
    Value multiples = tileOp.repeats();
    assert(isRankedShapedType(input.getType()) && "Expected Ranked ShapedType");
    ShapedType inputType = input.getType().cast<ShapedType>();
    Type elementType = inputType.getElementType();
    int64_t inputRank = inputType.getRank();
    SmallVector<Value, 4> inputShapeValues;
    Type indexType = rewriter.getI64Type();

    for (int64_t i = 0; i < inputRank; ++i) {
      int64_t dim_size = inputType.getDimSize(i);
      if (dim_size == ShapedType::kDynamicSize) {
        Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, input);
        Value dimSizeExtent =
            rewriter.create<shape::GetExtentOp>(loc, inputShape, i);
        Value dimSizeValue = rewriter.create<arith::IndexCastOp>(
            loc, RankedTensorType::get({1}, indexType), dimSizeExtent);
        inputShapeValues.push_back(dimSizeValue);
      } else {
        inputShapeValues.push_back(rewriter.create<mhlo::ConstantOp>(loc,
            DenseIntElementsAttr::get(RankedTensorType::get({1}, indexType),
                ArrayRef<int64_t>{dim_size})));
      }
    }

    RankedTensorType multiplesType =
        multiples.getType().dyn_cast<RankedTensorType>();
    Type multiplesElementType = multiplesType.getElementType();
    int64_t multiplesRank = multiplesType.getRank();
    if (multiplesRank != 1)
      return failure();
    if ((!multiplesType.hasStaticShape()) ||
        (multiplesType.getDimSize(0) != inputRank)) {
      return failure();
    }

    SmallVector<Value, 4> outDimSize;
    outDimSize.reserve(inputRank * 2);
    for (int64_t dim_idx = 0; dim_idx < inputRank; ++dim_idx) {
      Value multiples_size = rewriter.create<mhlo::SliceOp>(loc,
          RankedTensorType::get({1}, multiplesElementType), multiples,
          DenseIntElementsAttr::get(
              RankedTensorType::get({1}, multiplesElementType),
              ArrayRef<int64_t>{dim_idx}),
          DenseIntElementsAttr::get(
              RankedTensorType::get({1}, multiplesElementType),
              ArrayRef<int64_t>{dim_idx + 1}),
          DenseIntElementsAttr::get(
              RankedTensorType::get({1}, multiplesElementType),
              ArrayRef<int64_t>{1}));
      outDimSize.push_back(multiples_size);
      outDimSize.push_back(inputShapeValues[dim_idx]);
    }
    SmallVector<int64_t, 4> broadcastDimensions;
    broadcastDimensions.reserve(inputRank);
    for (int64_t dim_idx = 0; dim_idx < inputRank; ++dim_idx) {
      broadcastDimensions.push_back(1 + 2 * dim_idx);
    }
    DenseIntElementsAttr broadcast_dims_attr =
        rewriter.getI64VectorAttr(broadcastDimensions);

    Value out_dim_size_tensor = rewriter.create<mhlo::ConcatenateOp>(loc,
        RankedTensorType::get(
            {static_cast<int64_t>(outDimSize.size())}, indexType),
        outDimSize, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    SmallVector<int64_t, 4> broadcast_shape(
        inputRank * 2, ShapedType::kDynamicSize);
    RankedTensorType broadcast_type =
        RankedTensorType::get(broadcast_shape, elementType);
    Value broadcast = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, broadcast_type, input, out_dim_size_tensor, broadcast_dims_attr);

    // %shape = [MS1, MS2]
    SmallVector<Value, 4> shape_values;
    shape_values.reserve(inputRank);
    for (int64_t i = 0; i < inputRank; ++i) {
      Value dim_size_value = rewriter.create<mhlo::MulOp>(
          loc, outDimSize[2 * i], outDimSize[2 * i + 1]);
      shape_values.push_back(dim_size_value);
    }
    Value shape = rewriter.create<mhlo::ConcatenateOp>(loc,
        RankedTensorType::get(
            {static_cast<int64_t>(shape_values.size())}, indexType),
        shape_values, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    Value reshpaeOp = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, outputType, broadcast, shape);
    rewriter.replaceOp(op, reshpaeOp);
    return success();
  }
};

void populateLoweringONNXTileOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTileOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
