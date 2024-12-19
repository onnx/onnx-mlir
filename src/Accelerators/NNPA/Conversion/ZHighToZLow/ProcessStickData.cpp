/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- ProcessStickData.cpp - Process Stick data ----------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ZHigh operations to Krnl/Affine/SCF
// operations that operates on stickified input/output data.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ProcessStickData.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Conversion/ONNXToKrnl/Quantization/QuantizeHelper.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/SmallVectorHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

// Implementation of quantize helper function.
void emitDynamicQuantizationLinearMinMaxFromStickifiedInput(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    Value input, StringAttr inputLayout, Value &inputMin, Value &inputMax,
    bool enableSIMD, bool enableParallel) {
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MathBuilder, MemRefBuilder, VectorBuilder>;
  MDBuilder create(rewriter, loc);

  // Extract dims from input, set lbs/ubs.
  DimsExpr dims;
  create.krnlIE.getShapeAsSymbols(input, dims);
  int64_t rank = dims.size();
  IndexExpr zero = LitIE(0);
  DimsExpr lbs(rank, zero);
  DimsExpr ubs = dims;

  // Decide parameters.
  // UnrollVL decides how many vectors of 8 DLF16 will be processed at once.
  int64_t unrollVL = 4; // Experimentally good unroll factor.
  int64_t archVL = 8;   // DLF16.
  int64_t totVL = unrollVL * archVL;

  // If not parallel, threadNum = 1, forExplicitParallelLoopIE will simply pass
  // through the lb/ub, so ok to have parID = 0 for the sequential cases.
  int64_t parId = 0;
  int64_t threadNum = 1;
  if (enableParallel) {
    if (findSuitableParallelDimension(lbs, ubs, 0, rank - 1, parId, 8)) {
      threadNum = 8; // TODO use more flexible value.
      onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId],
          "simd min/max for DQL in parallel ");
    } else {
      enableParallel = false;
      onnxToKrnlParallelReport(
          op, false, -1, -1, "not enough work in simd min/max for DQL");
    }
  }

  // Alloc temp buffers (more when using parallel).
  Type f32Type = rewriter.getF32Type();
  // For each thread, we can use totVL temp values for the current min/max.
  // But to increase the compute ratio over mem, we will reuse the same tmp
  // memory location for a pair of totVL values being processed.
  int64_t tmpSizePerThread = totVL / 2; // Reduce pair in same tmp.
  int64_t tmpSize = threadNum * tmpSizePerThread;
  MemRefType redType = MemRefType::get({tmpSize}, f32Type);
  VectorType vec8xF32Type = VectorType::get({archVL}, f32Type);
  VectorType vec4xF32Type = VectorType::get({archVL / 2}, f32Type);

  Value minTmp = create.mem.alignedAlloc(redType);
  Value maxTmp = create.mem.alignedAlloc(redType);

  // Init min and max.
  Value minInit = create.math.positiveInf(f32Type);
  Value splatMinInit = create.vec.splat(vec8xF32Type, minInit);
  Value maxInit = create.math.negativeInf(f32Type);
  Value splatMaxInit = create.vec.splat(vec8xF32Type, maxInit);
  // Could parallelize init, here main thread do it all. Use SIMD of 8x.
  for (int64_t u = 0; u < tmpSize; u += 8) {
    IndexExpr offset = LitIE(u);
    create.vec.storeIE(splatMinInit, minTmp, {offset});
    create.vec.storeIE(splatMaxInit, maxTmp, {offset});
  }

  // Reduction into these temps.
  IndexExpr tNum = LitIE(threadNum);
  create.krnl.forExplicitParallelLoopIE(lbs[parId], ubs[parId], tNum,
      [&](const KrnlBuilder &ck, ValueRange loopInd) {
        IndexExprScope scope(ck);
        IndexExpr t = DimIE(loopInd[0]);
        DimsExpr currDims = SymListIE(dims);
        // Reduce lbs, ubs for parallel region, if any.
        DimsExpr currLbs = SymListIE(lbs);
        DimsExpr currUbs = SymListIE(ubs);
        // In sequential cases (threadNum ==1, loopInd[1,2]== orig lb,ub).
        currLbs[parId] = SymIE(loopInd[1]);
        currUbs[parId] = SymIE(loopInd[2]);
        // Cannot use krnl because we may not have affine bounds.
        SCFBuilder sb(ck);
        IterateOverStickInputData<SCFBuilder>(
            sb, op, currLbs, currUbs, currDims, inputLayout, input, nullptr,
            unrollVL, /*enableParallel*/ false,
            /*prefetch, disable as it causes issue with affine*/ false,
            [&](const KrnlBuilder &b, SmallVectorImpl<Value> &vecOf4xF32Vals,
                DimsExpr &loopIndices) {
              MDBuilder create(b);
              int64_t size = vecOf4xF32Vals.size();
              assert((size == 2 || size == 2 * unrollVL) && "unexpected size");
              // Since all threads share the same tmpMin/Max, needs to offset by
              // t * <size for one thread>.
              IndexExpr threadOffset = SymIE(t) * tmpSizePerThread;
              size = size / 2; // handle pairs of 2, so size=1 or unrollVL.
              for (int i = 0; i < size; ++i) {
                Value val0 = vecOf4xF32Vals[2 * i];
                Value val1 = vecOf4xF32Vals[2 * i + 1];
                // Load appropriate tmp, compute min/max, store in tmp.
                IndexExpr offset = threadOffset + LitIE(4 * i);
                Value currMin =
                    create.vec.loadIE(vec4xF32Type, minTmp, {offset});
                Value currMax =
                    create.vec.loadIE(vec4xF32Type, maxTmp, {offset});
                currMin = create.math.min(currMin, val0);
                currMax = create.math.max(currMax, val0);
                currMin = create.math.min(currMin, val1);
                currMax = create.math.max(currMax, val1);
                create.vec.storeIE(currMin, minTmp, {offset});
                create.vec.storeIE(currMax, maxTmp, {offset});
              }
            },
            [&](const KrnlBuilder &b, Value scalarF32Val,
                DimsExpr &loopIndices) {
              MDBuilder create(b);
              Value currMin = create.krnl.loadIE(minTmp, {zero});
              Value currMax = create.krnl.loadIE(maxTmp, {zero});
              currMin = create.math.min(currMin, scalarF32Val);
              currMax = create.math.max(currMax, scalarF32Val);
              create.krnl.storeIE(currMin, minTmp, {zero});
              create.krnl.storeIE(currMax, maxTmp, {zero});
            }); // Iterate over stick.
      });       // Explicit parallel loop (sequential if threadNum==1).

  // Now we have all the partial min/max inside the minTmp/maxTmp: reduce each
  // vectors with each others. Main thread reduces all the values. Use SIMD of
  // 8x.
  Value finalVecMin = create.vec.loadIE(vec8xF32Type, minTmp, {zero});
  Value finalVecMax = create.vec.loadIE(vec8xF32Type, maxTmp, {zero});
  for (int u = 8; u < tmpSize; u += 8) {
    IndexExpr offset = LitIE(u);
    Value currMin = create.vec.loadIE(vec8xF32Type, minTmp, {offset});
    Value currMax = create.vec.loadIE(vec8xF32Type, maxTmp, {offset});
    finalVecMin = create.math.min(finalVecMin, currMin);
    finalVecMax = create.math.max(finalVecMax, currMax);
  }

  // Horizontal reduction of the vectors into a scalar.
  inputMin = create.vec.reduction(VectorBuilder::MIN, finalVecMin);
  inputMax = create.vec.reduction(VectorBuilder::MAX, finalVecMax);
}

} // namespace onnx_mlir
