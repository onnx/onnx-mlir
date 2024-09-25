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

namespace onnx_mlir {

// Helper function for quantization.
void emitQuantizationLinearScalarParameters(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, MemRefType inputType, MemRefType quantizedType,
    Value alloc, DimsExpr &allocDims, Value input, Value qMin, Value qMax,
    Value scale, Value zeroPoint, bool hasZeroPoint, bool enableSIMD,
    bool enableParallel) {
  MultiDialectBuilder<KrnlBuilder, MemRefBuilder, VectorBuilder, MathBuilder>
      create(rewriter, loc);

  // Types
  Type quantizedElementType = quantizedType.getElementType();
  int64_t rank = inputType.getRank();

  // Flatten the input data and outputs
  DimsExpr inputDims, flatInputDims, flatAllocDims;
  inputDims = allocDims; // Unput and output have the same shape.
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
    GenOpMix mix = {{GenericOps::DivGop, 1}, {GenericOps::ArithmeticGop, 5},
        {GenericOps::ConversionGop, 1}, {GenericOps::MinMaxGop, 2},
        {GenericOps::MulGop, 2}, {GenericOps::SelectGop, 3},
        {GenericOps::FloorGop, 2},
        {GenericOps::EstimatedVectorRegisterPressure,
            8 /* Little parallelism in code. */}};
    totVL = computeSuitableUnrollFactor(inputType /* use unquantized type*/,
        innermostLoopCollapse, mix, canOverCompute, simdLoopStaticTripCount,
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

#if 1
  // hi alex: test with 2 loops for easier debugging
  // Allocate output buffers (same type as input).
  MemRefType flatBufferType = llvm::cast<MemRefType>(flatInput.getType());
  Value flatBuffer = create.mem.alignedAlloc(flatBufferType, flatInputDims);
  DimsExpr bufferAF;
  bufferAF.emplace_back(zero);

  create.krnl.simdIterateIE(simdLb, simdUb, totVL, simdOnly, enableParallel,
      {flatInput}, {inputAF}, {flatBuffer}, {bufferAF},
      {[&](const KrnlBuilder &kb, ArrayRef<Value> inputVals, int64_t VL) {
        MultiDialectBuilder<MathBuilder> create(kb);
        Value x = inputVals[0];
        // Scale
        Value scaleX = create.math.div(x, scale);
        // Round
        Value roundX = create.math.round(scaleX);
        // Adjust
        Value adjustX;
        if (hasZeroPoint)
          adjustX = create.math.add(roundX, zeroPoint);
        else
          adjustX = roundX;
        // Saturate: use max into a min.
        Value saturateX = create.math.clip(adjustX, qMin, qMax);
        // Old approach.
        // return create.math.cast(quantizedElementType, saturateX);
        return saturateX;
      }});

  // Need transient types.
  Type inputElementType = flatBufferType.getElementType();
  unsigned inputWidth;
  if (isa<Float32Type>(inputElementType))
    inputWidth = 32;
  else if (isa<Float64Type>(inputElementType))
    inputWidth = 64;
  else
    llvm_unreachable("unsupported input type");
  IntegerType quantizedIntType = cast<IntegerType>(quantizedElementType);
  bool isSignless = quantizedIntType.isSignless();
  bool isSigned = quantizedIntType.isSigned();
  Type quantizedElementTypeSameSizeAsInput =
      rewriter.getIntegerType(inputWidth, isSignless || isSigned);

  create.krnl.simdIterateIE(simdLb, simdUb, totVL, simdOnly, enableParallel,
      {flatBuffer}, {bufferAF}, {flatAlloc}, {outputAF},
      {[&](const KrnlBuilder &kb, ArrayRef<Value> inputVals, int64_t VL) {
        MultiDialectBuilder<KrnlBuilder, VectorBuilder, MathBuilder> create(kb);
        // Convert float* to int*/uint* where * is 32 (64?)
        Value input = inputVals[0];
        Value quantizedSameSizeAsInput =
            create.math.cast(quantizedElementTypeSameSizeAsInput, input);
    // Convert int32/uint32 to int*/unint* where * is 8, 16...
#if 0
        // Code get normalized to the code below
        unsigned quantizedWidth = quantizedIntType.getWidth();
        unsigned currWidth = inputWidth;
        Value qVal = quantizedSameSizeAsInput;
        while (currWidth > quantizedWidth) {
          currWidth = currWidth / 2;
          Type qType =
              rewriter.getIntegerType(currWidth, isSignless || isSigned);
          qVal = create.math.cast(qType, qVal);
        }
#else
        Value qVal =
            create.math.cast(quantizedElementType, quantizedSameSizeAsInput);
#endif
        return qVal;
      }});

#else
  // faster than original loop on z16, takes 124us for 64k vals
  // Allocate output buffers.
  MemRefType flatBufferType = llvm::cast<MemRefType>(flatInput.getType());
  Value flatBuffer = create.mem.alignedAlloc(flatBufferType, flatInputDims);
  DimsExpr bufferAF;
  bufferAF.emplace_back(zero);

  create.krnl.simdIterateIE(simdLb, simdUb, totVL, simdOnly, enableParallel,
      {flatInput}, {inputAF}, {flatBuffer}, {bufferAF},
      {[&](const KrnlBuilder &kb, ArrayRef<Value> inputVals, int64_t VL) {
        MultiDialectBuilder<MathBuilder> create(kb);
        Value x = inputVals[0];
        // Scale
        Value scaleX = create.math.div(x, scale);
        // Round
        Value roundX = create.math.round(scaleX);
        // Adjust
        Value adjustX;
        if (hasZeroPoint)
          adjustX = create.math.add(roundX, zeroPoint);
        else
          adjustX = roundX;
        // Saturate: use max into a min.
        Value saturateX = create.math.clip(adjustX, qMin, qMax);
        // Old approach.
        // return create.math.cast(quantizedElementType, saturateX);
        return saturateX;
      }});

  // A second loop that performs scalar float to int performs better than the
  // compiler's attempt to generate SIMD conversion code. This might not hold
  // with all data types, but is definitely noticeable with uint8.
  //
  // Investigate further: we might save the vector to a buffer on the fly
  // (avoiding a second loop as below), and then reload each value as scalar and
  // then saved them as scalar (thus avoiding the insert/extract SIMD operations
  // that also do not perform well). We can have a SIMD buffer in memory for the
  // non-quantized and quantized simd values, but then we also need to privatize
  // it, which is also not easy in this scheme. So ignore this for now.
  create.krnl.forLoopIE(simdLb, simdUb, 1, enableParallel,
      [&](const KrnlBuilder &kb, ValueRange loopInd) {
        MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(kb);
        Value buffVal = create.krnl.loadIE(flatBuffer, {zero}, {loopInd[0]});
        Value res = create.math.cast(quantizedElementType, buffVal);
        create.krnl.storeIE(res, flatAlloc, {zero}, {loopInd[0]});
      });
#endif

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
        zeroPoint, hasZeroPoint, enableSIMD, enableParallel);

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
