/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- DynamicQuantizeLinear.cpp - Lowering DynamicQuantizeLinear Op ----===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX DynamicQuantizeLinear Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXDynamicQuantizeLinearOpLowering
    : public OpConversionPattern<ONNXDynamicQuantizeLinearOp> {
  ONNXDynamicQuantizeLinearOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXDynamicQuantizeLinearOp dqlOp,
      ONNXDynamicQuantizeLinearOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    using LocalDialectBuilder = MultiDialectBuilder<KrnlBuilder,
        IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder, OnnxBuilder>;
    Operation *op = dqlOp.getOperation();
    Location loc = ONNXLoc<ONNXDynamicQuantizeLinearOp>(op);
    LocalDialectBuilder create(rewriter, loc);

    ValueRange operands = adaptor.getOperands();
    Value X = adaptor.getX();

    // MemRefType for inputs and outputs.
    auto xMemRefType = dyn_cast<MemRefType>(X.getType());
    auto yMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(dqlOp.getResult(0).getType()));
    auto yScaleMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(dqlOp.getResult(1).getType()));
    auto yZeroPointMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(dqlOp.getResult(2).getType()));

    // Types
    Type elementType = xMemRefType.getElementType();
    Type quantizedElementType = yMemRefType.getElementType();
    int64_t rank = xMemRefType.getRank();

    // Get shape.
    ONNXDynamicQuantizeLinearOpShapeHelper shapeHelper(
        op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Allocate output buffers.
    Value Y =
        create.mem.alignedAlloc(yMemRefType, shapeHelper.getOutputDims(0));
    Value YScale =
        create.mem.alignedAlloc(yScaleMemRefType, shapeHelper.getOutputDims(1));
    Value YZeroPoint = create.mem.alignedAlloc(
        yZeroPointMemRefType, shapeHelper.getOutputDims(2));

    // Equations:
    // y_scale = (max(x) - min(x))/(qmax - qmin)
    // intermediate_zero_point = qmin - min(x)/y_scale
    // y_zero_point = cast(round(saturate(itermediate_zero_point)))
    // y = saturate (round (x / y_scale) + y_zero_point)
    //
    // where, saturate is to clip to [0, 255] for ui8.

    // QMax, QMin.
    Value qMax = create.math.constant(elementType, 255.0);
    Value qMin = create.math.constant(elementType, 0.0);
    Value QMax = create.mem.alignedAlloc(yScaleMemRefType);
    create.krnl.store(qMax, QMax);
    Value QMin = create.mem.alignedAlloc(yScaleMemRefType);
    create.krnl.store(qMin, QMin);

    // Compute max(x) and min (x).
    Value none = create.onnx.none();
    Value XMax = create.onnx.toMemref(
        create.onnx.reduceMax(yScaleMemRefType, X, none, false));
    Value XMin = create.onnx.toMemref(
        create.onnx.reduceMin(yScaleMemRefType, X, none, false));
    Value xMax = create.krnl.load(XMax);
    Value xMin = create.krnl.load(XMin);
    // Include 0 to max(x) and min(x).
    // x_min = min(min(x), 0)
    // x_max = max(max(x), 0)
    Value zero = create.math.constant(elementType, 0.0);
    Value greaterThanZero = create.math.sgt(xMax, zero);
    xMax = create.math.select(greaterThanZero, xMax, zero);
    Value lessThanZero = create.math.slt(xMin, zero);
    xMin = create.math.select(lessThanZero, xMin, zero);

    // Compute y_scale.
    Value scale = create.math.div(
        create.math.sub(xMax, xMin), create.math.sub(qMax, qMin));
    create.krnl.store(scale, YScale);

    // Compute y_zero_point.
    Value interZeroPoint = create.math.div(create.math.sub(qMin, xMin), scale);
    // Saturate zero point.
    Value saturateZeroPoint =
        create.onnx.clip(interZeroPoint, qMin, qMax, /*scalarType=*/true);
    // Round zero point.
    Value zeroPoint = create.onnx.round(saturateZeroPoint, /*scalarType=*/true);
    Value zeroPointInt = create.math.cast(quantizedElementType, zeroPoint);
    create.krnl.store(zeroPointInt, YZeroPoint);

    // Compute y.
    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr> lbs(rank, LiteralIndexExpr(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.getOutputDims(0),
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder, OnnxBuilder> create(
              createKrnl);
          Value x = create.krnl.load(X, loopInd);
          // Scale
          Value scaleX = create.math.div(x, scale);
          // Round
          Value roundX = create.onnx.round(scaleX, /*scalarType=*/true);
          // Adjust
          Value adjustX = create.math.add(roundX, zeroPoint);
          // Saturate
          Value saturateX =
              create.onnx.clip(adjustX, qMin, qMax, /*scalarType=*/true);
          Value res = create.math.cast(quantizedElementType, saturateX);
          create.krnl.store(res, Y, loopInd);
        });

    rewriter.replaceOp(op, {Y, YScale, YZeroPoint});
    return success();
  }
};

void populateLoweringONNXDynamicQuantizeLinearOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXDynamicQuantizeLinearOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
