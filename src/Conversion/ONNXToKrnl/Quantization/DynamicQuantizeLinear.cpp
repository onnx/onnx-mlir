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
#include "src/Support/SmallVectorHelper.hpp"

#define HI_ALEX_NEW 1
#if HI_ALEX_NEW
// https://github.com/AlexandreEichenberger/onnx-mlir/pull/new/quant-opt-v1
#include "src/Compiler/CompilerOptions.hpp"
#endif

using namespace mlir;

namespace onnx_mlir {

using LocalDialectBuilder = MultiDialectBuilder<KrnlBuilder,
    IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder, OnnxBuilder>;

void emitDynamicQuantizationLinearScalarParameters(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    MemRefType inputType, MemRefType quantizedType, Value input, Value qMin,
    Value qMax, Value &scale, Value &zeroPoint, Value &quantizedZeroPoint) {
  LocalDialectBuilder create(rewriter, loc);

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
  Value saturateZeroPoint =
      create.onnx.clip(interZeroPoint, qMin, qMax, /*scalarType=*/true);
  // Round zero point.
  zeroPoint = create.onnx.round(saturateZeroPoint, /*scalarType=*/true);
  quantizedZeroPoint = create.math.cast(quantizedElementType, zeroPoint);
}

void emitSimdLoopIE(VectorBuilder &vb, IndexExpr ub, int64_t VL,
    llvm::ArrayRef<Value> inputs, llvm::ArrayRef<DimsExpr> inputAFs,
    llvm::ArrayRef<Value> outputs, llvm::ArrayRef<DimsExpr> outputAFs,
    bool fullySimd,
    function_ref<void(VectorBuilder &vb, llvm::ArrayRef<Value> inputVals,
        llvm::SmallVectorImpl<Value> &resVals)>
        bodyBuilderFn) {
  int64_t inputNum = inputs.size();
  assert(inputAFs.size() == inputs.size() && "expected same size");
  int64_t outputNum = outputs.size();
  assert(outputAFs.size() == outputs.size() && "expected same size");
  IndexExpr zero = LitIE(0);

  MultiDialectBuilder<KrnlBuilder, VectorBuilder> create(vb);

  // Full SIMD loops.
  IndexExpr lb = zero;
  if (VL > 1) {
    ValueRange loopDef = create.krnl.defineLoops(1);
    ValueRange blockedLoopDef = create.krnl.block(loopDef[0], VL);
    IndexExpr ubFullSimd = ub;
    if (!fullySimd)
      ub = ub - (VL - 1);
    create.krnl.iterateIE(loopDef, {blockedLoopDef[0]}, {zero}, {ubFullSimd},
        [&](KrnlBuilder &ck, ValueRange loopInd) {
          IndexExprScope scope(ck);
          MultiDialectBuilder<KrnlBuilder, VectorBuilder> create(ck);
          IndexExpr ind = DimIE(loopInd[0]);
          // Load all the inputs as vectors of VL values,
          llvm::SmallVector<Value, 4> vecInputVals;
          for (int64_t i = 0; i < inputNum; ++i) {
            MemRefType type = mlir::cast<MemRefType>(inputs[i].getType());
            VectorType vecType = VectorType::get({VL}, type.getElementType());
            DimsExpr AF = SymListIE(inputAFs[i]);
            int64_t rank = type.getRank();
            assert(rank == (int64_t)AF.size() && "AF expected input rank refs");
            AF[rank - 1] = AF[rank - 1] + ind;
            Value vecVal = create.vec.loadIE(vecType, inputs[i], AF, {});
            vecInputVals.emplace_back(vecVal);
          }
          // Call the method to compute the values.
          llvm::SmallVector<Value, 4> vecResVals;
          bodyBuilderFn(create.vec, vecInputVals, vecResVals);
          // Store all the outputs as vectors of VL values,
          for (int64_t i = 0; i < outputNum; ++i) {
            MemRefType type = mlir::cast<MemRefType>(outputs[i].getType());
            DimsExpr AF = SymListIE(outputAFs[i]);
            int64_t rank = type.getRank();
            assert(rank == (int64_t)AF.size() && "AF expected ouput rank refs");
            AF[rank - 1] = AF[rank - 1] + ind;
            create.vec.storeIE(vecResVals[i], outputs[i], AF, {});
          }
        });
    // Account for the loops performed there.
    // New lb = ub - ub % VL.
    IndexExpr ubModVL = ub % VL;
    if (fullySimd || ubModVL.isLiteralAndIdenticalTo(0)) {
      // Detect SIMD only.
      return;
    }
    lb = ub - ubModVL;
  }
  // Handle scalar values, if any.
  llvm_unreachable("not yet");
}

// TODO may consider SIMD and parallel.
struct ONNXDynamicQuantizeLinearOpLowering
    : public OpConversionPattern<ONNXDynamicQuantizeLinearOp> {
  ONNXDynamicQuantizeLinearOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

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

    // TODO: consider SIMD version of this.
    Value qMax = create.math.constant(elementType, 255.0);
    Value qMin = create.math.constant(elementType, 0.0);
    Value scale, zeroPoint, zeroPointInt;
    if (debugTestCompilerOpt) {
      emitDynamicQuantizationLinearScalarParameters(rewriter, loc, op,
          xMemRefType, yMemRefType, X, qMin, qMax, scale, zeroPoint,
          zeroPointInt);
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
    }
    create.krnl.store(scale, YScale);
    create.krnl.store(zeroPointInt, YZeroPoint);

    // Compute y.
    if (debugTestCompilerOpt) {
      ValueRange loopDef = create.krnl.defineLoops(rank - 1);
      SmallVector<IndexExpr, 4> lbs(rank - 1, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs =
          firstFew<IndexExpr, 4>(shapeHelper.getOutputDims(0), rank - 2);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            MultiDialectBuilder<KrnlBuilder, MathBuilder, VectorBuilder> create(
                createKrnl);
            IndexExpr simdUB = SymIE(shapeHelper.getOutputDims(0)[rank - 1]);
            int64_t VL = 4; // hi alex, refine this
            // Create access functions for input X and output Y.
            DimsExpr inputAF = SymListIE(loopInd);
            inputAF.emplace_back(LitIE(0));
            DimsExpr outputAF = SymListIE(loopInd);
            outputAF.emplace_back(LitIE(0));
            emitSimdLoopIE(create.vec, simdUB, VL, {X}, {inputAF}, {Y},
                {outputAF}, true,
                [&](VectorBuilder &vb, ArrayRef<Value> inputVals,
                    SmallVectorImpl<Value> &resVals) {
                  MultiDialectBuilder<MathBuilder, VectorBuilder> create(vb);
                  Value x = inputVals[0];
                  fprintf(stderr, "hi alex x, scale and zero point\n");
                  x.dump();
                  scale.dump();
                  zeroPoint.dump();
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

    } else {
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
