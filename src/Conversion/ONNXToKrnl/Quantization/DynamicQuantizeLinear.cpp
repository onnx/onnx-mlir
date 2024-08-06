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
#include "src/Conversion/ONNXToKrnl/Quantization/QuantizeHelper.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/SmallVectorHelper.hpp"

#define HI_ALEX_NEW 1
#if HI_ALEX_NEW
#include "src/Compiler/CompilerOptions.hpp"
#endif

using namespace mlir;

namespace onnx_mlir {

// Implementation of quantize helper function.
// TODO: enable/disable simd, add parallel.
void emitDynamicQuantizationLinearScalarParameters(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    MemRefType inputType, MemRefType quantizedType, Value input, Value qMin,
    Value qMax, Value &scale, Value &zeroPoint, Value &quantizedZeroPoint) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);

  // Types
  Type elementType = inputType.getElementType();
  Type quantizedElementType = quantizedType.getElementType();

  // Equations:
  // y_scale = (max(x) - min(x))/(qMax - qMin)
  // intermediate_zero_point = qMin - min(x)/y_scale
  // y_zero_point = cast(round(saturate(intermediate_zero_point)))
  // y = saturate (round (x / y_scale) + y_zero_point)
  //
  // where, saturate is to clip to [0, 255] for ui8.

  Value inputMinAlloc, inputMaxAlloc;
  emitMinMaxReductionToScalar(
      rewriter, loc, op, input, inputMinAlloc, inputMaxAlloc);
  Value xMin = create.krnl.load(inputMinAlloc);
  Value xMax = create.krnl.load(inputMaxAlloc);

  // Include 0 to max(x) and min(x).
  // x_min = min(min(x), 0)
  // x_max = max(max(x), 0)
  Value zero = create.math.constant(elementType, 0.0);
  xMax = create.math.max(xMax, zero);
  xMin = create.math.min(xMin, zero);
  // Compute y_scale.
  Value xDiff = create.math.sub(xMax, xMin);
  Value boundDiff = create.math.sub(qMax, qMin);
  scale = create.math.div(xDiff, boundDiff);

  // Compute y_zero_point.
  Value interZeroPoint = create.math.sub(qMin, create.math.div(xMin, scale));
  // Saturate zero point.
  Value saturateZeroPoint = create.math.clip(interZeroPoint, qMin, qMax);
  // Round zero point.
  zeroPoint = create.math.round(saturateZeroPoint);
  quantizedZeroPoint = create.math.cast(quantizedElementType, zeroPoint);
}

struct ONNXDynamicQuantizeLinearOpLowering
    : public OpConversionPattern<ONNXDynamicQuantizeLinearOp> {
  ONNXDynamicQuantizeLinearOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  // hi alex: scan what is still needed once old code is removed
  using LocalDialectBuilder = MultiDialectBuilder<KrnlBuilder,
      IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder, OnnxBuilder>;

  LogicalResult matchAndRewrite(ONNXDynamicQuantizeLinearOp dqlOp,
      ONNXDynamicQuantizeLinearOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
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

    Value qMax = create.math.constant(elementType, 255.0);
    Value qMin = create.math.constant(elementType, 0.0);
    Value scale, zeroPoint, zeroPointInt;

    if (debugTestCompilerOpt) {
      emitDynamicQuantizationLinearScalarParameters(rewriter, loc, op,
          xMemRefType, yMemRefType, X, qMin, qMax, scale, zeroPoint,
          zeroPointInt);
      create.krnl.store(scale, YScale);
      create.krnl.store(zeroPointInt, YZeroPoint);

      emitQuantizationLinearScalarParameters(rewriter, loc, op, xMemRefType,
          yMemRefType, Y, shapeHelper.getOutputDims(0), X, qMin, qMax, scale,
          zeroPoint);

    } else {
      // Equations:
      // y_scale = (max(x) - min(x))/(qmax - qmin)
      // intermediate_zero_point = qmin - min(x)/y_scale
      // y_zero_point = cast(round(saturate(itermediate_zero_point)))
      // y = saturate (round (x / y_scale) + y_zero_point)
      //
      // where, saturate is to clip to [0, 255] for ui8.

      // QMax, QMin.
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
      scale = create.math.div(
          create.math.sub(xMax, xMin), create.math.sub(qMax, qMin));

      // Compute y_zero_point.
      Value interZeroPoint =
          create.math.sub(qMin, create.math.div(xMin, scale));
      // Saturate zero point.
      Value saturateZeroPoint =
          create.onnx.clip(interZeroPoint, qMin, qMax, /*scalarType=*/true);
      // Round zero point.
      zeroPoint = create.onnx.round(saturateZeroPoint, /*scalarType=*/true);
      zeroPointInt = create.math.cast(quantizedElementType, zeroPoint);

      create.krnl.store(scale, YScale);
      create.krnl.store(zeroPointInt, YZeroPoint);

      ValueRange loopDef = create.krnl.defineLoops(rank);
      SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
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
    }

    rewriter.replaceOp(op, {Y, YScale, YZeroPoint});
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXDynamicQuantizeLinearOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXDynamicQuantizeLinearOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
