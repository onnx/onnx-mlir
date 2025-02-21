/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-------- ProcessStickData.hpp.inc - Process Stick data --------------===//
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

namespace onnx_mlir {

// Iterate over each stick, for an original size of dims, and cover the
// iterations from lbs to ubs. In most cases, lbs={0...} and ubs=dims, namely we
// cover all iterations. But we can parallelize the loops from the outside, in
// which case we expect lbs and ubs to reflect the iterations assigned to this
// thread. Note that we cannot tile in the innermost dim (as this is the
// dimension of the sticks).
template <class BUILDER>
void IterateOverStickInputData(const BUILDER &b, mlir::Operation *op,
    DimsExpr &lbs, DimsExpr &ubs, DimsExpr &dims, mlir::StringAttr inputLayout,
    mlir::Value input, mlir::Value output, int64_t unrollVL,
    bool enableParallel, bool enablePrefetch,
    ContiguousVectorOfF32IterateBodyFn processVectorOfF32Vals,
    ScalarF32IterateBodyFn processScalarF32Val) {
  // Init builder and scopes.
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MemRefBuilder, VectorBuilder, SCFBuilder>;
  MDBuilder create(b);
  //IndexExprScope initialScope(b);
  // Get info and check some inputs.
  int64_t rank = dims.size();
  int64_t d1 = rank - 1;
  IndexExpr E1 = dims[d1];
  assert(lbs.size() == ubs.size() && "expected same sizes");
  assert(lbs.size() == dims.size() && "expected same sizes");
  assert((inputLayout.getValue().equals_insensitive("4D") ||
             inputLayout.getValue().equals_insensitive("3D") ||
             inputLayout.getValue().equals_insensitive("2D") ||
             inputLayout.getValue().equals_insensitive("3DS") ||
             inputLayout.getValue().equals_insensitive("NHWC")) &&
         "unsupported inputLayout");

  // Info for SIMD Vector Length (VL).
  int64_t archVL = 8;              // FP16 archVL.
  int64_t archVLHalf = archVL / 2; // FP32 archVL.
  int64_t totVL = archVL * unrollVL;
  int64_t stickLen = 64;
  assert(stickLen % totVL == 0 && "bad unrollVL factor");
  mlir::Type f16Type = b.getBuilder().getF16Type();
  mlir::Type f32Type = b.getBuilder().getF32Type();
  mlir::VectorType vecF16Type = mlir::VectorType::get({archVL}, f16Type);
  mlir::MemRefType bufferF32Type = mlir::MemRefType::get({archVL}, f32Type);

  // Useful constants.
  IndexExpr litZero = LitIE(0);
  IndexExpr lit1 = LitIE(1);
  IndexExpr lit2 = LitIE(2);
  IndexExpr litArchVLHalf = LitIE(archVLHalf);
  IndexExpr litArchVL = LitIE(archVL);
  IndexExpr litStickLen = LitIE(stickLen);

  // Create loop iterations. We iterate over E1 as sticks of 64 elements. Lbs
  // and ubs reflect the iteration over the sticks (tiled data points).
  DimsExpr tiledLbs = lbs;
  DimsExpr tiledUbs = ubs;
  tiledUbs[d1] = E1.ceilDiv(litStickLen);

  // Predicates used to avoid creating code that is never used.
  bool neverHas64 = E1.isLiteralAndSmallerThan(stickLen);
  bool neverHas8 = E1.isLiteralAndSmallerThan(archVL);
  bool hasOnly64 = E1.isLiteral() && (E1.getLiteral() % stickLen == 0);
  bool hasOnly8 = E1.isLiteral() && (E1.getLiteral() % archVL == 0);

  // Parallel... Should not be turned on when parallelized in the outside.
  int64_t parId = 0;
  if (enableParallel) {
    // TODO: may want to check if ub of rank makes sense here.
    // Its ok here even to partition rank-1, included in (0..rank(, because
    // rank-1 is tiled. So we are still dealing with multiple of sticks.
    if (findSuitableParallelDimension(tiledLbs, tiledUbs, 0, rank, parId, 8)) {
      onnxToKrnlParallelReport(op, true, parId, tiledLbs[parId],
          tiledUbs[parId], "compiler-generated stickify");
    } else {
      enableParallel = false;
      onnxToKrnlParallelReport(op, false, -1, -1,
          "no dim with enough work in compiler-generated stickify");
    }
  }

  // Compute max sticks (tiles of 64 values). It is actually not easy to compute
  // the max number of sticks. Since we don't allocate, it is just a "view", we
  // only need to index by the "stick size", it is sufficient to assume 2 or
  // more.
  DimsExpr reallocStickDims = {lit2, litStickLen};
  mlir::Value inputAsSticks =
      create.mem.reinterpretCast(input, reallocStickDims);

  llvm::SmallVector<int64_t, 4> steps(rank, 1);
  llvm::SmallVector<bool, 4> useParallel(rank, false);
  if (enableParallel)
    useParallel[parId] = true;
  b.forLoopsIE(tiledLbs, tiledUbs, steps, useParallel,
      [&](const BUILDER &b, mlir::ValueRange tiledLoopInd) {
        MDBuilder create(b);
        IndexExprScope outerScope(b);
        DimsExpr tiledOuterIndices = DimListIE(tiledLoopInd);
        // Computation for accessing data (not tiled, actual indices).
        DimsExpr outerIndices = tiledOuterIndices;
        IndexExpr E1 = SymIE(dims[d1]); // Original upper bound in d1.
        IndexExpr e1 = outerIndices[d1] = tiledOuterIndices[d1] * litStickLen;
        // Translate the tile index t1 to the actual targetted data. Have to
        // give the actual indices, not the tiled ones.
        mlir::Value inputOffset =
            create.krnl.getLinearOffsetIndexIE(input, outerIndices);
        // Offset in inputAsSticks's first dim is as multiple of litStickLen, so
        // divide by it.
        IndexExpr inputStickOffset = SymIE(inputOffset).floorDiv(litStickLen);
        // Buffer for small leftovers (used when E1 % 8 != 0)
        mlir::Value bufferF32;
        if (!hasOnly8)
          bufferF32 = create.mem.alignedAlloc(bufferF32Type);
        if (enablePrefetch) {
          // Prefetch current line
          create.krnl.prefetchIE(input, outerIndices, /*write*/ false,
              /*locality*/ 1);
          if (output)
            create.krnl.prefetchIE(output, outerIndices, /*write*/ true,
                /*locality*/ 1);
        }
        // Check if we have a full stick (aka end of stick is not beyond UB).
        IndexExpr hasFullStick;
        if (hasOnly64) {
          hasFullStick = PredIE(true); // Has only full sicks.
        } else if (neverHas64) {
          hasFullStick = PredIE(false); // Doesn't even has 1 stick.
        } else {
          IndexExpr isFull = create.krnlIE.isTileFull(e1, litStickLen, E1);
          hasFullStick = (isFull >= 0);
        }
        create.scf.ifThenElse(
            hasFullStick.getValue(),
            // If is full.
            [&](const SCFBuilder b) {
              if (neverHas64)
                return; // Nothing to do here. Avoid generating dead code.
              MDBuilder create(b);
              // Iterate through stick by totVL (aka 8 * unroll).
              create.scf.forLoopIE(litZero, litStickLen, totVL, /*par*/ false,
                  [&](const SCFBuilder b, mlir::ValueRange loopInd) {
                    MDBuilder create(b);
                    IndexExprScope innerScope(b, &outerScope);
                    IndexExpr l = DimIE(loopInd[0]);
                    DimsExpr innerIndices = SymListIE(outerIndices);
                    innerIndices[d1] = innerIndices[d1] + l;
                    mlir::SmallVector<mlir::Value, 8> vecOfF32Vals;
                    // Load archVL (8) f16 values from input via reinterpreted
                    // data tile, and then convert them into f32 and enqueue in
                    // vecOfF32Vals.
                    for (int64_t u = 0; u < unrollVL; ++u) {
                      mlir::Value vecOfF16 =
                          create.vec.loadIE(vecF16Type, inputAsSticks,
                              {SymIE(inputStickOffset), l + (u * archVL)});
                      auto convertOp =
                          b.getBuilder()
                              .create<zlow::ZLowConvertDLF16ToF32VectorOp>(
                                  b.getLoc(), vecOfF16);
                      vecOfF32Vals.emplace_back(convertOp.getResult(0));
                      vecOfF32Vals.emplace_back(convertOp.getResult(1));
                    }
                    processVectorOfF32Vals(
                        create.krnl, vecOfF32Vals, innerIndices);
                  });
            },
            // Else, we don't have a full (64 e1) tile.
            [&](SCFBuilder b) {
              if (hasOnly64)
                return; // Do not generate dead code.
              MDBuilder create(b);
              IndexExprScope middleScope(b, &outerScope);
              IndexExpr tripCount = SymIE(E1) - SymIE(e1);
              if (!neverHas8) {
                // Note: if we only have multiple of VL, loop below will
                // handle all as we subtract (VL-1). Aka if VL=8 and tripCount
                // = 16, tripCountWithoutPartialLastVL is 16 - 7 = 9. Thus we
                // iterate over i=0 & i=8 as both are < 9.
                IndexExpr tripCountWithoutPartialLastVL =
                    tripCount - (archVL - 1);
                create.scf.forLoopIE(litZero, tripCountWithoutPartialLastVL,
                    archVL, /*par*/ false,
                    [&](SCFBuilder b, mlir::ValueRange loopInd) {
                      MDBuilder create(b);
                      IndexExprScope innerScope(b, &middleScope);
                      IndexExpr l = DimIE(loopInd[0]);
                      DimsExpr innerIndices = SymListIE(outerIndices);
                      innerIndices[d1] = innerIndices[d1] + l;
                      mlir::SmallVector<mlir::Value, 8> vecOfF32Vals;
                      // Load f16 values from input via reinterpreted data
                      // tile.
                      mlir::Value vecOfF16 = create.vec.loadIE(vecF16Type,
                          inputAsSticks, {SymIE(inputStickOffset), l});
                      // Convert back to f32.
                      auto convertOp =
                          b.getBuilder()
                              .create<zlow::ZLowConvertDLF16ToF32VectorOp>(
                                  b.getLoc(), vecOfF16);
                      vecOfF32Vals.emplace_back(convertOp.getResult(0));
                      vecOfF32Vals.emplace_back(convertOp.getResult(1));
                      processVectorOfF32Vals(
                          create.krnl, vecOfF32Vals, innerIndices);
                    });
              }
              if (!hasOnly8) {
                // Deal with the last <8 values: compute f32 using simd.
                IndexExpr remainingScalarValues = tripCount % archVL;
                IndexExpr lastL = tripCount - remainingScalarValues;
                mlir::Value vecOfF16 = create.vec.loadIE(vecF16Type,
                    inputAsSticks, {SymIE(inputStickOffset), lastL});
                // Convert back to f32.
                auto convertOp =
                    b.getBuilder().create<zlow::ZLowConvertDLF16ToF32VectorOp>(
                        b.getLoc(), vecOfF16);
                mlir::Value vecF32H = convertOp.getResult(0);
                mlir::Value vecF32L = convertOp.getResult(1);
                // Save into archVL value buffer.
                create.vec.storeIE(vecF32H, bufferF32, {litZero});
                create.vec.storeIE(vecF32L, bufferF32, {litArchVLHalf});
                create.scf.forLoopIE(litZero, remainingScalarValues, 1,
                    /*par*/ false, [&](SCFBuilder b, mlir::ValueRange loopInd) {
                      MDBuilder create(b);
                      IndexExprScope innerScope(b, &middleScope);
                      IndexExpr l = DimIE(loopInd[0]);
                      // Load converted value.
                      mlir::Value f32 = create.krnl.loadIE(bufferF32, {l});

                      DimsExpr innerIndices = SymListIE(outerIndices);
                      innerIndices[d1] = innerIndices[d1] + SymIE(lastL);
                      innerIndices[d1] = innerIndices[d1] + l;
                      processScalarF32Val(create.krnl, f32, innerIndices);
                    });
              }
            });
      });
}

} // namespace onnx_mlir
