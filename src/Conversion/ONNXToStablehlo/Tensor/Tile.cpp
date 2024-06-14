/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Tile.cpp - Lowering Tile Op----------------------=== //
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Tile Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

// ONNXTileOp(A) is mainly implemented using Stablehlo broadcastOp & reshapeOp
struct ONNXTileOpLoweringToStablehlo : public ConversionPattern {
  ONNXTileOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTileOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTileOpAdaptor operandAdaptor(operands);
    ONNXTileOp tileOp = cast<ONNXTileOp>(op);
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();

    // I believe it is not currently used.
    IndexExprBuilderForAnalysis createIE(loc);
    ONNXTileOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");

    Value input = tileOp.getInput();
    Value multiples = tileOp.getRepeats();
    assert(isRankedShapedType(input.getType()) && "Expected Ranked ShapedType");
    ShapedType inputType = mlir::cast<ShapedType>(input.getType());
    Type elementType = inputType.getElementType();
    int64_t inputRank = inputType.getRank();
    SmallVector<Value, 4> inputShapeValues;
    Type indexType = rewriter.getI64Type();

    for (int64_t i = 0; i < inputRank; ++i) {
      int64_t dim_size = inputType.getDimSize(i);
      if (dim_size == ShapedType::kDynamic) {
        Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, input);
        Value dimSizeExtent =
            rewriter.create<shape::GetExtentOp>(loc, inputShape, i);
        Value dimSizeValue = rewriter.create<arith::IndexCastOp>(
            loc, RankedTensorType::get({1}, indexType), dimSizeExtent);
        inputShapeValues.push_back(dimSizeValue);
      } else {
        inputShapeValues.push_back(rewriter.create<stablehlo::ConstantOp>(
            loc, DenseElementsAttr::get(RankedTensorType::get({1}, indexType),
                     ArrayRef<int64_t>{dim_size})));
      }
    }

    RankedTensorType multiplesType =
        mlir::dyn_cast<RankedTensorType>(multiples.getType());
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
      Value multiples_size = rewriter.create<stablehlo::SliceOp>(loc,
          RankedTensorType::get({1}, multiplesElementType), multiples,
          DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{dim_idx}),
          DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{dim_idx + 1}),
          DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1}));
      outDimSize.push_back(multiples_size);
      outDimSize.push_back(inputShapeValues[dim_idx]);
    }
    SmallVector<int64_t, 4> broadcastDimensions;
    broadcastDimensions.reserve(inputRank);
    for (int64_t dim_idx = 0; dim_idx < inputRank; ++dim_idx) {
      broadcastDimensions.push_back(1 + 2 * dim_idx);
    }
    DenseI64ArrayAttr broadcast_dims_attr =
        rewriter.getDenseI64ArrayAttr(broadcastDimensions);

    Value out_dim_size_tensor = rewriter.create<stablehlo::ConcatenateOp>(loc,
        RankedTensorType::get(
            {static_cast<int64_t>(outDimSize.size())}, indexType),
        outDimSize, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    SmallVector<int64_t, 4> broadcast_shape(
        inputRank * 2, ShapedType::kDynamic);
    RankedTensorType broadcast_type =
        RankedTensorType::get(broadcast_shape, elementType);
    Value broadcast = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
        loc, broadcast_type, input, out_dim_size_tensor, broadcast_dims_attr);

    // %shape = [MS1, MS2]
    SmallVector<Value, 4> shape_values;
    shape_values.reserve(inputRank);
    for (int64_t i = 0; i < inputRank; ++i) {
      Value dim_size_value = rewriter.create<stablehlo::MulOp>(
          loc, outDimSize[2 * i], outDimSize[2 * i + 1]);
      shape_values.push_back(dim_size_value);
    }
    Value shape = rewriter.create<stablehlo::ConcatenateOp>(loc,
        RankedTensorType::get(
            {static_cast<int64_t>(shape_values.size())}, indexType),
        shape_values, IntegerAttr::get(rewriter.getIntegerType(64), 0));
    Value reshpaeOp = rewriter.create<stablehlo::DynamicReshapeOp>(
        loc, outputType, broadcast, shape);
    rewriter.replaceOp(op, reshpaeOp);
    return success();
  }
};

void populateLoweringONNXTileOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTileOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
