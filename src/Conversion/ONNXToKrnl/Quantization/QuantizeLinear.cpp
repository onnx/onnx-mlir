/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- QuantizeLinear.cpp - Lowering QuantizeLinear Op -------------===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX QuantizeLinear Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/SmallVectorHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

// Helper function for quantization.
void emitQuantizationLinearScalarParameters(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, MemRefType inputType, MemRefType quantizedType,
    Value alloc, DimsExpr &allocDims, Value input, Value qMin, Value qMax,
    Value scale, Value zeroPoint, bool enableSIMD, bool enableParallel) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);

  // Types
  Type quantizedElementType = quantizedType.getElementType();
  int64_t rank = inputType.getRank();

  // Determine a suitable SIMD vector length for this loop.
  int64_t totVL = 1;
  int64_t simdLoopStaticTripCount = 0;
  if (enableSIMD) {
    totVL = VectorBuilder::computeSuitableUnrollFactor(
        inputType /* use unquantized type*/,
        1 /* only innermost loop is simdized */,
        {GenericOps::DivGop, GenericOps::ArithmeticGop,
            GenericOps::ConversionGop, GenericOps::MinMaxGop,
            GenericOps::MulGop, GenericOps::SelectGop, GenericOps::FloorGop},
        {1, 5, 1, 2, 2, 3, 2}, simdLoopStaticTripCount);
  }
  // Has only simd iterations when we have SIMD (totVL > 1), the simd dimensions
  // is a multiple of a non-zero constant (simdLoopStaticTripCount) iterations,
  // and simdLoopStaticTripCount % totVL == 0.
  bool onlySimdIterations = (simdLoopStaticTripCount > 0) && (totVL > 1) &&
                            (simdLoopStaticTripCount % totVL == 0);

  // Generate outer loops
  ValueRange loopDef = create.krnl.defineLoops(rank - 1);
  SmallVector<IndexExpr, 4> lbs(rank - 1, LitIE(0));
  SmallVector<IndexExpr, 4> ubs = allocDims;
  ubs.pop_back(); // Remove the last dim.
  IndexExpr zero = LitIE(0);
  create.krnl.iterateIE(
      loopDef, loopDef, lbs, ubs, [&](KrnlBuilder &kb, ValueRange loopInd) {
        IndexExprScope scope(kb);
        MultiDialectBuilder<KrnlBuilder, MathBuilder, VectorBuilder> create(kb);
        IndexExpr simdLb = zero;
        IndexExpr simdUb = SymIE(allocDims[rank - 1]);
        // Create access functions for input X and output Y.
        DimsExpr inputAF = SymListIE(loopInd);
        inputAF.emplace_back(zero);
        DimsExpr outputAF = SymListIE(loopInd);
        outputAF.emplace_back(zero);
        create.krnl.simdIterateIE(simdLb, simdUb, totVL, onlySimdIterations,
            {input}, {inputAF}, {alloc}, {outputAF},
            [&](KrnlBuilder &kb, ArrayRef<Value> inputVals,
                SmallVectorImpl<Value> &resVals) {
              MultiDialectBuilder<MathBuilder> create(kb);
              Value x = inputVals[0];
              // Scale
              Value scaleX = create.math.div(x, scale);
              // Round
              Value roundX = create.math.round(scaleX);
              // Adjust
              Value adjustX = create.math.add(roundX, zeroPoint);
              // Saturate
              Value saturateX = create.math.clip(adjustX, qMin, qMax);
              Value res = create.math.cast(quantizedElementType, saturateX);
              resVals.emplace_back(res);
            });
      });
  if (totVL > 1)
    onnxToKrnlSimdReport(op, /*successful*/ true, totVL,
        simdLoopStaticTripCount, "quantizationLinear whole tensor");
  else
    onnxToKrnlSimdReport(op, /*successful*/ false, 0, 0,
        "no simd in quantizationLinear whole tensor");
}

struct ONNXQuantizeLinearOpLowering
    : public OpConversionPattern<ONNXQuantizeLinearOp> {
  ONNXQuantizeLinearOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableSIMD, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx), enableSIMD(enableSIMD),
        enableParallel(enableParallel) {}

  bool enableSIMD = false;
  bool enableParallel = false;

  using LocalDialectBuilder = MultiDialectBuilder<KrnlBuilder,
      IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder>;

  LogicalResult matchAndRewrite(ONNXQuantizeLinearOp qlOp,
      ONNXQuantizeLinearOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = qlOp.getOperation();
    Location loc = ONNXLoc<ONNXQuantizeLinearOp>(op);
    LocalDialectBuilder create(rewriter, loc);

    ValueRange operands = adaptor.getOperands();
    Value X = adaptor.getX();
    Value YScale = adaptor.getYScale();
    Value YZeroPoint = qlOp.getYZeroPoint(); // Optional input.

    // MemRefType for inputs and outputs.
    auto xMemRefType = dyn_cast<MemRefType>(X.getType());
    auto yMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(qlOp.getResult().getType()));
    MemRefType yScaleMemRefType = mlir::cast<MemRefType>(YScale.getType());

    // Types
    Type elementType = xMemRefType.getElementType();
    Type quantizedElementType = yMemRefType.getElementType();

    // Does not support per-axis and i8.
    assert(yScaleMemRefType.getRank() == 0 &&
           "Does not support per-axis quantization");
    assert(quantizedElementType.isUnsignedInteger() &&
           "Does not support i8 quantization");

    // Get shape.
    ONNXQuantizeLinearOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Allocate output buffers.
    Value Y =
        create.mem.alignedAlloc(yMemRefType, shapeHelper.getOutputDims(0));

    // Equations:
    // y = saturate (round (x / y_scale) + y_zero_point)
    //
    // where, saturate is to clip to [0, 255] for ui8 or [-128, 127] it's i8.

    // Quantization bounds.
    Value qMax, qMin;
    if (quantizedElementType.isUnsignedInteger()) {
      qMax = create.math.constant(elementType, 255.0);
      qMin = create.math.constant(elementType, 0.0);
    } else {
      qMax = create.math.constant(elementType, 127.0);
      qMin = create.math.constant(elementType, -128.0);
    }

    // Load y_scale.
    Value scale = create.krnl.load(YScale);

    // Load y_zero_point.
    Value zeroPoint;
    if (!isNoneValue(YZeroPoint)) {
      zeroPoint = create.krnl.load(adaptor.getYZeroPoint());
      zeroPoint = create.math.cast(elementType, zeroPoint);
    } else
      zeroPoint = create.math.constant(elementType, 0.0);

    emitQuantizationLinearScalarParameters(rewriter, loc, op, xMemRefType,
        yMemRefType, Y, shapeHelper.getOutputDims(0), X, qMin, qMax, scale,
        zeroPoint, enableSIMD, enableParallel);

    rewriter.replaceOp(op, {Y});
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXQuantizeLinearOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD,
    bool enableParallel) {
  patterns.insert<ONNXQuantizeLinearOpLowering>(
      typeConverter, ctx, enableSIMD, enableParallel);
}

} // namespace onnx_mlir
