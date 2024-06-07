/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- OneHot.cpp - Lowering OneHot Op -------------------===//
//
// Copyright 2023-2024
//
// =============================================================================
//
// This file lowers the ONNX OneHot Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#include <numeric>

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXOneHotOpLoweringToStablehlo
    : public OpConversionPattern<ONNXOneHotOp> {
  ONNXOneHotOpLoweringToStablehlo(MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(ONNXOneHotOp onehotOp,
      ONNXOneHotOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = onehotOp.getOperation();
    MLIRContext *context = op->getContext();
    Location loc = ONNXLoc<ONNXOneHotOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value indices = adaptor.getIndices();
    Value depthValue = adaptor.getDepth();
    Value values = adaptor.getValues();
    Type outputType = *op->result_type_begin();

    IndexExprBuilderForStablehlo createIE(rewriter, loc);
    ONNXOneHotOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    int64_t axis = shapeHelper.axis;

    RankedTensorType indicesType =
        mlir::dyn_cast<RankedTensorType>(indices.getType());
    if (!indicesType || !indicesType.hasStaticShape())
      return failure();
    ArrayRef<int64_t> indicesShape = indicesType.getShape();
    Type indicesElementType = indicesType.getElementType();

    DenseIntElementsAttr depthAttr;
    if (!matchPattern(depthValue, m_Constant(&depthAttr))) {
      return failure();
    }

    int64_t depth = depthAttr.getValues<APInt>()[0].getSExtValue();

    llvm::SmallVector<int64_t, 4> broadcastDims(indicesShape.size());
    std::iota(broadcastDims.begin(), broadcastDims.begin() + axis, 0);
    std::iota(broadcastDims.begin() + axis, broadcastDims.end(), axis + 1);

    llvm::SmallVector<int64_t, 4> outputDims = llvm::to_vector<4>(indicesShape);
    outputDims.insert(outputDims.begin() + axis, depth);

    RankedTensorType indexType =
        RankedTensorType::get(llvm::ArrayRef(outputDims), indicesElementType);

    Value iota = rewriter.create<stablehlo::IotaOp>(
        loc, indexType, IntegerAttr::get(rewriter.getIntegerType(64), axis));
    Value broadcastIndices = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, indexType, indices, rewriter.getDenseI64ArrayAttr(broadcastDims));
    Value zero = rewriter.create<stablehlo::ConstantOp>(loc,
        DenseIntElementsAttr::get(RankedTensorType::get({}, indicesElementType),
            ArrayRef<int64_t>{0}));
    Value broadcastZero = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, indexType, zero, rewriter.getDenseI64ArrayAttr({}));
    Value broadcastDepth;
    int64_t depthRank =
        mlir::cast<RankedTensorType>(depthValue.getType()).getRank();
    if (depthRank == 1)
      broadcastDepth = rewriter.create<stablehlo::BroadcastInDimOp>(
          loc, indexType, depthValue, rewriter.getDenseI64ArrayAttr({0}));
    else
      broadcastDepth = rewriter.create<stablehlo::BroadcastInDimOp>(
          loc, indexType, depthValue, rewriter.getDenseI64ArrayAttr({}));
    Value compareGeZero = rewriter.create<stablehlo::CompareOp>(loc,
        broadcastIndices, broadcastZero, stablehlo::ComparisonDirection::GE);
    Value positiveIndices = rewriter.create<stablehlo::AddOp>(
        loc, broadcastIndices, broadcastDepth);
    Value normalizedIndices = rewriter.create<stablehlo::SelectOp>(
        loc, indexType, compareGeZero, broadcastIndices, positiveIndices);
    Value compare = rewriter.create<stablehlo::CompareOp>(
        loc, normalizedIndices, iota, stablehlo::ComparisonDirection::EQ);
    Type valueType = mlir::cast<ShapedType>(values.getType()).getElementType();
    Value offValue = rewriter.create<stablehlo::SliceOp>(loc,
        RankedTensorType::get({1}, valueType), values,
        DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{0}),
        DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1}),
        DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1}));
    Value onValue = rewriter.create<stablehlo::SliceOp>(loc,
        RankedTensorType::get({1}, valueType), values,
        DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1}),
        DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{2}),
        DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1}));
    Value offValueBroadcast = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, outputType, offValue, rewriter.getDenseI64ArrayAttr({0}));
    Value onValueBroadcast = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, outputType, onValue, rewriter.getDenseI64ArrayAttr({0}));
    Value result = rewriter.create<stablehlo::SelectOp>(
        loc, outputType, compare, onValueBroadcast, offValueBroadcast);
    rewriter.replaceOp(op, {result});
    return success();
  }
};

} // namespace

void populateLoweringONNXOneHotOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXOneHotOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
