/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- DynamicQuantizeLinear.cpp - Lowering DynamicQuantizeLinear Op ----===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX DynamicQuantizeLinear Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Conversion/ONNXToKrnl/Quantization/QuantizeHelper.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/SmallVectorHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

// Implementation of computing the min and max over an entire tensor. Can
// generate parallel and SIMD code as requested.
void emitDynamicQuantizationLinearMinMax(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Value input, Value &inputMin, Value &inputMax,
    bool enableSIMD, bool enableParallel) {
  MultiDialectBuilder<KrnlBuilder> create(rewriter, loc);
  Value inputMinAlloc, inputMaxAlloc;
  emitMinMaxReductionToScalar(rewriter, loc, op, input, inputMinAlloc,
      inputMaxAlloc, enableSIMD, enableParallel);
  inputMin = create.krnl.load(inputMinAlloc);
  inputMax = create.krnl.load(inputMaxAlloc);
}

// Implementation of quantize helper function. Returns Values scale, zeroPoint,
// and quantizedZeroPoint directly (not as a value to a memory location,
// directly the floating point results).
void emitDynamicQuantizationLinearScalarParametersFromMinMax(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    MemRefType inputType, MemRefType quantizedType, Value inputMin,
    Value inputMax, Value qMin, Value qMax, Value &scale, Value &zeroPoint,
    Value &quantizedZeroPoint, bool wantZeroPoint, bool enableParallel) {

  // Equations:
  // y_scale = (max(x) - min(x))/(qMax - qMin)
  // intermediate_zero_point = qMin - min(x)/y_scale
  // y_zero_point = cast(round(saturate(intermediate_zero_point)))
  // y = saturate (round (x / y_scale) + y_zero_point)
  //
  // where, saturate is to clip to [0, 255] for ui8.

  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
  // Types.
  Type elementType = inputType.getElementType();
  Type quantizedElementType = quantizedType.getElementType();
  // Include 0 to max(x) and min(x).
  // x_min = min(min(x), 0)
  // x_max = max(max(x), 0)
  Value zero = create.math.constant(elementType, 0.0);
  inputMax = create.math.max(inputMax, zero);
  inputMin = create.math.min(inputMin, zero);
  // Compute y_scale.
  Value xDiff = create.math.sub(inputMax, inputMin);
  Value boundDiff = create.math.sub(qMax, qMin);
  scale = create.math.div(xDiff, boundDiff);

  // Compute y_zero_point.
  if (wantZeroPoint) {
    Value interZeroPoint =
        create.math.sub(qMin, create.math.div(inputMin, scale));
    // Saturate zero point.
    Value saturateZeroPoint = create.math.clip(interZeroPoint, qMin, qMax);
    // Round zero point.
    zeroPoint = create.krnl.roundEven(saturateZeroPoint);
  } else {
    zeroPoint = zero;
  }
  quantizedZeroPoint = create.math.cast(quantizedElementType, zeroPoint);
}

struct ONNXDynamicQuantizeLinearOpLowering
    : public OpConversionPattern<ONNXDynamicQuantizeLinearOp> {
  ONNXDynamicQuantizeLinearOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, bool enableSIMD, bool enableParallel,
      bool enableFastMath)
      : OpConversionPattern(typeConverter, ctx), enableSIMD(enableSIMD),
        enableParallel(enableParallel), enableFastMath(enableFastMath) {}

  bool enableSIMD = false;
  bool enableParallel = false;
  bool enableFastMath = false;

  using LocalDialectBuilder = MultiDialectBuilder<KrnlBuilder,
      IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder>;

  LogicalResult matchAndRewrite(ONNXDynamicQuantizeLinearOp dqlOp,
      ONNXDynamicQuantizeLinearOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = dqlOp.getOperation();
    Location loc = ONNXLoc<ONNXDynamicQuantizeLinearOp>(op);
    LocalDialectBuilder create(rewriter, loc);

    ValueRange operands = adaptor.getOperands();
    Value X = adaptor.getX();

    // MemRefType for inputs and outputs.
    auto xMemRefType = mlir::dyn_cast<MemRefType>(X.getType());
    auto yMemRefType = mlir::dyn_cast<MemRefType>(
        typeConverter->convertType(dqlOp.getResult(0).getType()));
    auto yScaleMemRefType = mlir::dyn_cast<MemRefType>(
        typeConverter->convertType(dqlOp.getResult(1).getType()));
    auto yZeroPointMemRefType = mlir::dyn_cast<MemRefType>(
        typeConverter->convertType(dqlOp.getResult(2).getType()));

    // Types
    Type elementType = xMemRefType.getElementType();

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

    Value xMin, xMax;
    emitDynamicQuantizationLinearMinMax(
        rewriter, loc, op, X, xMin, xMax, enableSIMD, enableParallel);
    Value qMax = create.math.constant(elementType, 255.0);
    Value qMin = create.math.constant(elementType, 0.0);
    Value scale, zeroPoint, zeroPointInt;
    bool wantZeroPoint = !disableQuantZeroPoint;
    emitDynamicQuantizationLinearScalarParametersFromMinMax(rewriter, loc, op,
        xMemRefType, yMemRefType, xMin, xMax, qMin, qMax, scale, zeroPoint,
        zeroPointInt, wantZeroPoint, enableParallel);
    create.krnl.store(scale, YScale);
    create.krnl.store(zeroPointInt, YZeroPoint);

    emitQuantizationLinearScalarParameters(rewriter, loc, op, xMemRefType,
        yMemRefType, Y, shapeHelper.getOutputDims(0), X, qMin, qMax, scale,
        zeroPoint, wantZeroPoint /*wanted one, so we have a zero point*/,
        enableSIMD, enableParallel, enableFastMath);

    rewriter.replaceOp(op, {Y, YScale, YZeroPoint});
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXDynamicQuantizeLinearOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx,
    bool enableSIMD, bool enableParallel, bool enableFastMath) {
  patterns.insert<ONNXDynamicQuantizeLinearOpLowering>(
      typeConverter, ctx, enableSIMD, enableParallel, enableFastMath);
}

} // namespace onnx_mlir
