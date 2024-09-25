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

  Type inputElementType = inputType.getElementType();
  unsigned inputWidth;
  if (isa<Float32Type>(inputElementType))
    inputWidth = 32;
  else if (isa<Float64Type>(inputElementType))
    inputWidth = 64;
  else
    llvm_unreachable("unsupported input type");
  IntegerType quantizedIntType = cast<IntegerType>(quantizedElementType);
  bool isSigned = quantizedIntType.isSignless() || quantizedIntType.isSigned();
  Type quantizedElementTypeInputSized;
  if (isSigned) {
    // Cannot use getIntegerType(inputWidth, true) as it returns signed ints.
    if (inputWidth == 64)
      quantizedElementTypeInputSized = rewriter.getI64Type();
    else if (inputWidth == 32)
      quantizedElementTypeInputSized = rewriter.getI32Type();
    else
      llvm_unreachable("unsupported input type");
  } else {
    // unsigned of the right type
    quantizedElementTypeInputSized = rewriter.getIntegerType(inputWidth, false);
  }
  create.krnl.simdIterateIE(simdLb, simdUb, totVL, simdOnly, enableParallel,
      {flatInput}, {inputAF}, {flatAlloc}, {outputAF},
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
        // Convert float* to int*/uint* where * is 64/32.
        Value qSaturateXInputSized =
            create.math.cast(quantizedElementTypeInputSized, saturateX);
        // Reduce quantized precision.
        Value res =
            create.math.cast(quantizedElementType, qSaturateXInputSized);
        return res;
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
