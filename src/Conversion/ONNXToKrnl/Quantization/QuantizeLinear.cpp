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

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/SmallVectorHelper.hpp"

using namespace mlir;

#define DISABLE_FAST_MATH 0 /* disable reciprocal (for debug) */

namespace onnx_mlir {

// Helper function for quantization.
void emitQuantizationLinearScalarParameters(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, MemRefType inputType, MemRefType quantizedType,
    Value alloc, DimsExpr &allocDims, Value input, Value qMin, Value qMax,
    Value scale, Value zeroPoint, bool hasZeroPoint, bool enableSIMD,
    bool enableParallel, bool enableFastMath) {
  MultiDialectBuilder<KrnlBuilder, MemRefBuilder, VectorBuilder, MathBuilder>
      create(rewriter, loc);

  // Types
  Type quantizedElementType = quantizedType.getElementType();
  Type inputElementType = inputType.getElementType();
  int64_t rank = inputType.getRank();

  // Use fast math with reciprocal?
  bool useReciprocal =
      !DISABLE_FAST_MATH && enableFastMath && isa<FloatType>(inputElementType);

  // Flatten the input data and outputs
  DimsExpr inputDims, flatInputDims, flatAllocDims;
  inputDims = allocDims; // Unput and output have the same shape.
                         //
  if (rank == 0) {
    // Do scalar computation only when the input is a scalar tensor.
    Value x = create.krnl.load(input);
    // Scale
    Value scaleX;
    if (useReciprocal) {
      Value one = create.math.constant(inputElementType, 1.0);
      Value scaleReciprocal = create.math.div(one, scale);
      scaleX = create.math.mul(x, scaleReciprocal);
    } else {
      scaleX = create.math.div(x, scale);
    }
    // Round
    Value roundX = create.krnl.roundEven(scaleX);
    // Adjust
    Value adjustX;
    if (hasZeroPoint)
      adjustX = create.math.add(roundX, zeroPoint);
    else
      adjustX = roundX;
    // Saturate: use max into a min.
    Value saturateX = create.math.clip(adjustX, qMin, qMax);
    // Convert into quantized type.
    Value quantSaturateX = create.math.cast(quantizedElementType, saturateX);
    create.krnl.store(quantSaturateX, alloc);
    onnxToKrnlSimdReport(op, /*successful*/ false, 0, 0,
        "no simd in quantizationLinear whole tensor");
    return;
  }

  Value flatInput =
      create.mem.reshapeToFlatInnermost(input, inputDims, flatInputDims, rank);
  Value flatAlloc =
      create.mem.reshapeToFlatInnermost(alloc, allocDims, flatAllocDims, rank);

  // Determine a suitable SIMD vector length for this loop.
  int64_t totVL = 1;
  int64_t simdLoopStaticTripCount = 0;
  bool simdOnly = false;
  if (enableSIMD) {
    int64_t innermostLoopCollapse = 1; // Only innermost is simdized.
    bool canOverCompute = false;
    GenOpMix mixAdjust;
    if (hasZeroPoint)
      mixAdjust = {{GenericOps::ArithmeticGop, 1}};
    GenOpMix mixRound = getGenOpMix<ONNXRoundOp>(inputElementType, op);
    GenericOps divOrMulGenOp =
        useReciprocal ? GenericOps::MulGop : GenericOps::DivGop;
    GenOpMix mixOthers = {{divOrMulGenOp, 1}, {GenericOps::ConversionGop, 1},
        {GenericOps::MinMaxGop, 2},
        {GenericOps::EstimatedVectorRegisterPressure, 4}};
    GenOpMix mix1 = computeGenOpMixUnion(mixAdjust, mixRound);
    GenOpMix mix2 = computeGenOpMixUnion(mix1, mixOthers);
    totVL = computeSuitableUnrollFactor(inputType /* use unquantized type*/,
        innermostLoopCollapse, mix2, canOverCompute, simdLoopStaticTripCount,
        simdOnly);
  }

  IndexExpr zero = LitIE(0);
  IndexExpr simdLb = zero;
  IndexExpr simdUb = flatAllocDims[0];
  // Create access functions for input X and output Y.
  DimsExpr inputAF;
  inputAF.emplace_back(zero);
  DimsExpr outputAF;
  outputAF.emplace_back(zero);

  Value scaleReciprocal;
  if (useReciprocal) {
    Value one = create.math.constant(inputElementType, 1.0);
    scaleReciprocal = create.math.div(one, scale);
  }
  create.krnl.simdIterateIE(simdLb, simdUb, totVL, simdOnly, enableParallel,
      {flatInput}, {inputAF}, {flatAlloc}, {outputAF},
      {[&](const KrnlBuilder &kb, ArrayRef<Value> inputVals, int64_t VL) {
        MultiDialectBuilder<KrnlBuilder, MathBuilder> create(kb);
        Value x = inputVals[0];
        // Scale
        Value scaleX;
        if (useReciprocal)
          scaleX = create.math.mul(x, scaleReciprocal);
        else
          scaleX = create.math.div(x, scale);
        // Round
        Value roundX = create.krnl.roundEven(scaleX);
        // Adjust
        Value adjustX;
        if (hasZeroPoint)
          adjustX = create.math.add(roundX, zeroPoint);
        else
          adjustX = roundX;
        // Saturate: use max into a min.
        Value saturateX = create.math.clip(adjustX, qMin, qMax);
        // Convert into quantized type.
        return create.math.cast(quantizedElementType, saturateX);
      }});

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
      bool enableSIMD, bool enableParallel, bool enableFastMath)
      : OpConversionPattern(typeConverter, ctx), enableSIMD(enableSIMD),
        enableParallel(enableParallel), enableFastMath(enableFastMath) {}

  bool enableSIMD = false;
  bool enableParallel = false;
  bool enableFastMath = false;

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
    auto xMemRefType = mlir::dyn_cast<MemRefType>(X.getType());
    auto yMemRefType = mlir::dyn_cast<MemRefType>(
        typeConverter->convertType(qlOp.getResult().getType()));
    MemRefType yScaleMemRefType = mlir::cast<MemRefType>(YScale.getType());

    // Types
    Type elementType = xMemRefType.getElementType();
    Type quantizedElementType = yMemRefType.getElementType();

    // Does not support per-axis and other types rather than i8.
    assert(yScaleMemRefType.getRank() == 0 &&
           "Does not support per-axis quantization");
    assert(quantizedElementType.isInteger(8) &&
           "Only support i8/ui8 quantization at this moment");

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
    bool hasZeroPoint = false;
    if (!isNoneValue(YZeroPoint)) {
      zeroPoint = create.krnl.load(adaptor.getYZeroPoint());
      zeroPoint = create.math.cast(elementType, zeroPoint);
      hasZeroPoint = true;
    }
    if (disableQuantZeroPoint) {
      // TODO: should we expect to disable hasZeroPoint forcefully, or
      // generate an error if we had a zero point? Right now, just forcefully
      // assert we have no zero point, i.e. ignore one even if we had a zero
      // point.
      hasZeroPoint = false;
    }
    emitQuantizationLinearScalarParameters(rewriter, loc, op, xMemRefType,
        yMemRefType, Y, shapeHelper.getOutputDims(0), X, qMin, qMax, scale,
        zeroPoint, hasZeroPoint, enableSIMD, enableParallel, enableFastMath);

    rewriter.replaceOp(op, {Y});
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXQuantizeLinearOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD,
    bool enableParallel, bool enableFastMath) {
  patterns.insert<ONNXQuantizeLinearOpLowering>(
      typeConverter, ctx, enableSIMD, enableParallel, enableFastMath);
}

} // namespace onnx_mlir
