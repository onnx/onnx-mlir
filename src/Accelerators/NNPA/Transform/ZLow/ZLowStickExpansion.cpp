/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- ZLowStickExpansion.cpp - ZLow Stick/Unstick Expansion Patterns ---===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// This pass implements optimizations for ZLow operations, by substituting calls
// to stick / unstick with explict code to perform the transformation, when
// applicable.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

#include <map>

#define DEBUG_TYPE "zlow-stick-expansion"

// Todo: cleanup after we are done experimenting.
#define ENABLE_CSU_PAR true /* Allow parallel compiler gen Stick/Unstick. */
#define PREFETCH_CSU_DIST 0
#define PREFETCH_CSU 1

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

using MDBuilder = MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder,
    MathBuilder, MemRefBuilder, VectorBuilder, AffineBuilder, SCFBuilder>;

/// Expand unstick operation to compiler generated code for suitable patterns,
/// aka all but the 1D and 2DS data layouts at this time.
class UnstickExpansionPattern : public OpRewritePattern<ZLowUnstickOp> {
public:
  UnstickExpansionPattern(MLIRContext *context, bool enableParallelism = false)
      : OpRewritePattern<ZLowUnstickOp>(context, 1),
        enableParallel(enableParallelism) {}

  bool enableParallel = true;

  using OpRewritePattern<ZLowUnstickOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ZLowUnstickOp unstickOp, PatternRewriter &rewriter) const override {

    // Generic way to handle all formats listed below.
    // Did not add the HWCK as this is typically for constants and want to
    // preserve the high level constant propagation of constant values into the
    // Convolution filters.
    StringAttr layout = unstickOp.getLayoutAttr();
    if (layout.getValue().equals_insensitive("4D") ||
        layout.getValue().equals_insensitive("3D") ||
        layout.getValue().equals_insensitive("2D") ||
        layout.getValue().equals_insensitive("3DS") ||
        (layout.getValue().equals_insensitive("NHWC"))) {
      return generateUnstickCodeNoBuffer(rewriter, unstickOp);
    }
    // Otherwise, we don't replace and keep the zdnn call.
    return failure();
  }

  // The only requirement for this code to generate the proper code is that E1
  // is been sticked by 64.
  LogicalResult generateUnstickCodeNoBuffer(
      PatternRewriter &rewriter, ZLowUnstickOp unstickOp) const {
    Operation *op = unstickOp.getOperation();
    Location loc = unstickOp.getLoc();
    MDBuilder create(rewriter, loc);
    IndexExprScope allocScope(create.krnl);

    // Compute output dims and rank.
    Value input = unstickOp.getX();
    Value alloc = unstickOp.getOut();
    DimsExpr outputDims;
    create.krnlIE.getShapeAsSymbols(alloc, outputDims);
    int64_t rank = outputDims.size();

    // Info for SIMD Vector Length (VL) and associated types.
    int64_t archVL = 8;              // FP16 archVL.
    int64_t archVLHalf = archVL / 2; // FP32 archVL.
    assert(64 % archVL == 0 && "SIMD vector length must divide 64");
    Type f16Type = rewriter.getF16Type();
    Type f32Type = rewriter.getF32Type();
    VectorType vecF16Type = VectorType::get({archVL}, f16Type);
    MemRefType bufferType = MemRefType::get({archVL}, f32Type);

    // Define useful literals.
    IndexExpr litZero = LitIE(0);
    IndexExpr lit1 = LitIE(1);
    IndexExpr litArchVLHalf = LitIE(archVLHalf);
    IndexExpr litArchVL = LitIE(archVL);
    IndexExpr lit64 = LitIE(64);

    // Useful references for indexing dimensions (neg val are not used).
    int64_t E1 = rank - 1;

    // Create loop iterations. Note that we iterate over E1 as tiles of 64
    // elements.
    ValueRange loopDefs = create.krnl.defineLoops(rank);
    DimsExpr lbs(rank, litZero);
    DimsExpr ubs = outputDims;
    IndexExpr T1 = outputDims[E1].ceilDiv(64);
    ubs[E1] = T1; // E1 dim is over tiles.

    // Parallel...
    if (enableParallel) {
      int64_t parId;
      // TODO: may want to check if ub of rank makes sense here.
      if (findSuitableParallelDimension(lbs, ubs, 0, rank, parId, 8)) {
        create.krnl.parallel(loopDefs[parId]);
        onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId],
            "compiler-generated stickify");
      } else {
        onnxToKrnlParallelReport(op, false, -1, -1,
            "no dim with enough work in compiler-generated stickify");
      }
    }

    // Compute max tiles. It is actually not easy to compute the max number of
    // tiles. Since we don't allocate, it is just a "view", we only need to
    // index by the "tile size", it is sufficient to assume 2 or more. Tiles are
    // 64.
    IndexExpr T = LitIE(2);
    DimsExpr reallocTileDims = {T, lit64};
    Value inputAsTx64 =
        create.mem.reinterpretCast(input, litZero.getValue(), reallocTileDims);

    // Outer loop (E4, E3, E2, E1 iterates over tiles of 64 elements)
    create.krnl.iterateIE(
        loopDefs, loopDefs, lbs, ubs, [&](KrnlBuilder &b, ValueRange loopInd) {
          MDBuilder create(b);
          IndexExprScope outerScope(create.krnl, &allocScope);
          DimsExpr outerIndices = DimListIE(loopInd);
          // Computation for reading inputs.
          DimsExpr inputAF = outerIndices;
          IndexExpr e1 = outerIndices[E1] * 64;
          inputAF[E1] = e1;
          // Translate the tile index t1 to the actual targetted data.
          Value inputOffset =
              create.krnl.getLinearOffsetIndexIE(input, inputAF);
          IndexExpr inputDataOffset = SymIE(inputOffset);
          IndexExpr inputTileOffset = inputDataOffset.floorDiv(64);

// Prefetch
#if PREFETCH_CSU
          DimsExpr prefetchAF = inputAF;
          // Prefetch current line
          create.krnl.prefetchIE(input, prefetchAF, /*isWrite*/ false,
              /*locality*/ 1);
          create.krnl.prefetchIE(alloc, prefetchAF, /*isWrite*/ true,
              /*locality*/ 1);
#if PREFETCH_CSU_DIST > 0
          // Prefetch line in advance.
          prefetchAF[E1] = prefetchAF[E1] + (PREFETCH_CSU_DIST * 64);
          create.krnl.prefetchIE(input, prefetchAF, /*isWrite*/ false,
              /*locality*/ 1);
          create.krnl.prefetchIE(alloc, prefetchAF, /*isWrite*/ true,
              /*locality*/ 1);
#endif
#endif

          // I may process here up to [e1 ... e1 + m*64), make sure its
          // not going out of bound, i.e. beyond outputDIms[E1];
          IndexExpr ub1 = SymIE(outputDims[E1]);
          IndexExpr lit64Bis = LitIE(64);
          IndexExpr isFull = create.krnlIE.isTileFull(e1, lit64, ub1);
          IndexExpr isFullLogical = isFull >= 0;
          create.scf.ifThenElse(
              // Condition
              isFullLogical.getValue(),
              // Then (is full).
              [&](SCFBuilder b) {
                MDBuilder create(b);
                // Loop (tried unroll of 2 and 8, 4 was best).
                const int64_t unrollVL = 4;
                const int64_t totVL = unrollVL * archVL;
                assert(totVL <= 64 && "bad unroll");
                create.scf.forLoop(litZero.getValue(), lit64.getValue(), totVL,
                    [&](SCFBuilder b, Value loopIndex) {
                      MDBuilder create(b);
                      IndexExprScope innerScope(b, &outerScope);
                      IndexExpr l = DimIE(loopIndex);
                      Value vecF16[unrollVL], vecF32H[unrollVL],
                          vecF32L[unrollVL];
                      // Load f16 values from input via reinterpreted data tile.
                      for (int64_t i = 0; i < unrollVL; ++i) {
                        vecF16[i] = create.vec.loadIE(vecF16Type, inputAsTx64,
                            {SymIE(inputTileOffset), l + (i * archVL)}, {});
                      }
                      // Convert back to f32.
                      for (int64_t i = 0; i < unrollVL; ++i) {
                        auto convertOp =
                            rewriter.create<ZLowConvertDLF16ToF32VectorOp>(
                                loc, vecF16[i]);
                        vecF32H[i] = convertOp.getResult(0);
                        vecF32L[i] = convertOp.getResult(1);
                      }
                      // Store f32 values back to the (normal layout) output.
                      DimsExpr outputAF = SymListIE(inputAF);
                      outputAF[E1] = outputAF[E1] + l;
                      for (int64_t i = 0; i < unrollVL; ++i) {
                        LitIE iH(i * archVL), iL(i * archVL + archVL / 2);
                        create.vec.storeIE(
                            vecF32H[i], alloc, outputAF, {iH.getValue()});
                        create.vec.storeIE(
                            vecF32L[i], alloc, outputAF, {iL.getValue()});
                      }
                    });
              },
              // else, we don't have a full (64 e1) tile.
              [&](SCFBuilder b) {
                MDBuilder create(b);
                IndexExprScope middleScope(b, &outerScope);
                IndexExpr tripCount = SymIE(ub1) - SymIE(e1);
                // Note: if we only have multiple of VL, loop below will handle
                // all as we subtract (VL-1). Aka if VL=8 and tripCount = 16,
                // tripCountWithoutPartialLastVL is 16 - 7 = 9. Thus we iterate
                // over i=0 & i=8 as both are < 9.
                IndexExpr tripCountWithoutPartialLastVL =
                    tripCount - (archVL - 1);
                create.scf.forLoop(litZero.getValue(),
                    tripCountWithoutPartialLastVL.getValue(), archVL,
                    [&](SCFBuilder b, Value loopIndex) {
                      MDBuilder create(b);
                      IndexExprScope innerScope(b, &middleScope);
                      IndexExpr l = DimIE(loopIndex);
                      // Load f16 values from input via reinterpreted data tile.
                      Value vecF16 = create.vec.loadIE(vecF16Type, inputAsTx64,
                          {SymIE(inputTileOffset), l}, {});
                      // Convert back to f32.
                      auto convertOp =
                          rewriter.create<ZLowConvertDLF16ToF32VectorOp>(
                              loc, vecF16);
                      Value vecF32H = convertOp.getResult(0);
                      Value vecF32L = convertOp.getResult(1);
                      // Store f32 values back to the (normal layout) output.
                      DimsExpr outputAF = SymListIE(inputAF);
                      outputAF[E1] = outputAF[E1] + l;
                      create.vec.storeIE(vecF32H, alloc, outputAF, {});
                      create.vec.storeIE(
                          vecF32L, alloc, outputAF, {litArchVLHalf.getValue()});
                    });
                // Deal with the last values: compute f32 using simd.
                IndexExpr remainingScalarValues = tripCount % archVL;
                IndexExpr lastL = tripCount - remainingScalarValues;
                Value vecF16 = create.vec.loadIE(vecF16Type, inputAsTx64,
                    {SymIE(inputTileOffset), lastL}, {});
                // Convert back to f32.
                auto convertOp =
                    rewriter.create<ZLowConvertDLF16ToF32VectorOp>(loc, vecF16);
                Value vecF32H = convertOp.getResult(0);
                Value vecF32L = convertOp.getResult(1);
                // Save into archVL value buffer.
                Value bufferF32 = create.mem.alignedAlloca(bufferType);
                create.vec.storeIE(vecF32H, bufferF32, {litZero}, {});
                create.vec.storeIE(vecF32L, bufferF32, {litArchVLHalf}, {});
                // Save the remaining values as scalars.
                create.scf.forLoop(litZero.getValue(),
                    remainingScalarValues.getValue(), 1,
                    [&](SCFBuilder b, Value loopIndex) {
                      MDBuilder create(b);
                      IndexExprScope innerScope(b, &middleScope);
                      IndexExpr l = DimIE(loopIndex);
                      // Load converted value.
                      Value f32 = create.krnl.loadIE(bufferF32, {l});
                      DimsExpr outputAF = SymListIE(inputAF);
                      outputAF[E1] = outputAF[E1] + SymIE(lastL);
                      outputAF[E1] = outputAF[E1] + l;
                      create.krnl.storeIE(f32, alloc, outputAF);
                    });
              });
        });
    rewriter.eraseOp(unstickOp);
    return success();
  }
};

/// Expand stick operation to compiler generated code for suitable patterns, aka
/// all but the 1D and 2DS data layouts at this time.
class StickExpansionPattern : public OpRewritePattern<ZLowStickOp> {
public:
  StickExpansionPattern(MLIRContext *context, bool enableParallelism = false)
      : OpRewritePattern<ZLowStickOp>(context, 1),
        enableParallel(enableParallelism) {}

  bool enableParallel;

  using OpRewritePattern<ZLowStickOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ZLowStickOp stickOp, PatternRewriter &rewriter) const override {

    StringAttr layout = stickOp.getLayoutAttr();

    // Generic way to handle all formats listed below.
    // Did not add the HWCK as this is typically for constants and want to
    // preserve the high level constant propagation of constant values into the
    // Convolution filters.
    if (layout.getValue().equals_insensitive("4D") ||
        layout.getValue().equals_insensitive("3D") ||
        layout.getValue().equals_insensitive("2D") ||
        layout.getValue().equals_insensitive("3DS") ||
        layout.getValue().equals_insensitive("NHWC")) {
      return generateStickCodeNoBuffer(rewriter, stickOp);
    }
    // Otherwise, we don't replace and keep the zdnn call.
    return failure();
  }

  // Version without buffer, more like zdnn.
  // The only requirement for this code to generate the proper code is that E1
  // is been sticked by 64.
  LogicalResult generateStickCodeNoBuffer(
      PatternRewriter &rewriter, ZLowStickOp stickOp) const {
    Operation *op = stickOp.getOperation();
    Location loc = stickOp.getLoc();
    MDBuilder create(rewriter, loc);
    IndexExprScope allocScope(create.krnl);

    // Compute output dims and rank.
    Value input = stickOp.getX();
    Value alloc = stickOp.getOut();
    std::optional<int64_t> saturationOpt = stickOp.getSaturation();
    bool saturation = saturationOpt.has_value() && saturationOpt.value() != 0;

    DimsExpr outputDims;
    create.krnlIE.getShapeAsSymbols(alloc, outputDims);
    int64_t rank = outputDims.size();

    // Info for SIMD Vector Length (VL) and associated types.
    int64_t archVL = 8;              // FP16 archVL.
    int64_t archVLHalf = archVL / 2; // FP32 archVL.
    assert(64 % archVL == 0 && "SIMD vector length must divide 64");
    Type f32Type = rewriter.getF32Type();
    VectorType vecF32Type = VectorType::get({archVLHalf}, f32Type);

    // Define useful literals.
    IndexExpr litZero = LitIE(0);
    IndexExpr lit1 = LitIE(1);
    IndexExpr lit64 = LitIE(64);

    // Values for saturation.
    Value vecDlf16Min, vecDlf16Max;
    if (saturation) {
      Value dlf16Min = create.math.constant(f32Type, DLF16_MIN);
      vecDlf16Min = create.vec.splat(vecF32Type, dlf16Min);
      Value dlf16Max = create.math.constant(f32Type, DLF16_MAX);
      vecDlf16Max = create.vec.splat(vecF32Type, dlf16Max);
    }

    // Useful references for indexing dimensions (neg val are not used).
    int64_t E1 = rank - 1;

    // Create loop iterations. Note that we iterate over E1 as tiles of 64
    // elements.
    ValueRange loopDefs = create.krnl.defineLoops(rank);
    DimsExpr lbs(rank, litZero);
    DimsExpr ubs = outputDims;
    IndexExpr T1 = outputDims[E1].ceilDiv(64);
    ubs[E1] = T1; // E1 dim is over tiles.

    // If outputDims[E1] is constant and < 64, then T1 is 1 (ok), and we can
    // iterate over fewer values in the SIMD loop.
    IndexExpr simdLoopUB = lit64;
    int64_t unrollVL = 4; // Unrolling of SIMD loop: tried 2 and 8, 4 was best.
    if (outputDims[E1].isLiteral()) {
      int64_t d1 = outputDims[E1].getLiteral();
      if (d1 < 64) {
        // Shrink unrollVL if suitable.
        if (d1 <= archVL)
          unrollVL = 1;
        else if (d1 <= 2 * archVL)
          unrollVL = 2;
        else if (d1 <= 3 * archVL)
          unrollVL = 3;
        double trip = unrollVL * archVL;
        int64_t ub = std::ceil((1.0 * d1) / trip) * trip;
        simdLoopUB = LitIE(ub);
      }
    }
    int64_t totVL = unrollVL * archVL;
    assert(totVL <= 64 && "bad unroll");

    // Parallel...
    if (enableParallel) {
      int64_t parId;
      // TODO: may want to check if ub of rank makes sense here.
      if (findSuitableParallelDimension(lbs, ubs, 0, rank, parId, 8)) {
        create.krnl.parallel(loopDefs[parId]);
        onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId],
            "compiler-generated stickify");
      } else {
        onnxToKrnlParallelReport(op, false, -1, -1,
            "no dim with enough work in compiler-generated stickify");
      }
    }

    // Compute max tiles. It is actually not easy to compute the max number of
    // tiles. Since we don't allocate, it is just a "view", we only need to
    // index by the "tile size", it is sufficient to assume 2 or more. Tiles are
    // 64 elements.
    IndexExpr T = LitIE(2);
    DimsExpr reallocTileDims = {T, lit64};
    Value allocAsTx64 =
        create.mem.reinterpretCast(alloc, litZero.getValue(), reallocTileDims);

    // Outer loop (E1 iterates over tiles of 64 elements).
    create.krnl.iterateIE(
        loopDefs, loopDefs, lbs, ubs, [&](KrnlBuilder &b, ValueRange loopInd) {
          MDBuilder create(b);
          IndexExprScope outerScope(create.krnl, &allocScope);
          DimsExpr outerIndices;
          getIndexExprList<SymIE>(loopInd, outerIndices);
          DimsExpr memAF = outerIndices;
          memAF[E1] = memAF[E1] * 64; // Loop index for E1 is in tiles of 64.
          Value allocOffset = create.krnl.getLinearOffsetIndexIE(alloc, memAF);
          IndexExpr allocTileIndex = SymIE(allocOffset).floorDiv(64);
#if PREFETCH_CSU
          DimsExpr prefetchAF = memAF;
          // Prefetch current lines.
          create.krnl.prefetchIE(input, prefetchAF, /*isWrite*/ false,
              /*locality*/ 1);
          create.krnl.prefetchIE(alloc, prefetchAF, /*isWrite*/ true,
              /*locality*/ 1);
#if PREFETCH_CSU_DIST > 0
          // Prefetch line in advance.
          prefetchAF[E1] = prefetchAF[E1] + (PREFETCH_CSU_DIST * 64);
          create.krnl.prefetchIE(input, prefetchAF, /*isWrite*/ false,
              /*locality*/ 1);
          create.krnl.prefetchIE(alloc, prefetchAF, /*isWrite*/ true,
              /*locality*/ 1);
#endif
#endif

          create.affine.forIE(litZero, simdLoopUB, totVL,
              [&](AffineBuilder &b, ValueRange loopInd) {
                MDBuilder create(b);
                DimsExpr inputAF;
                IndexExprScope innerScope(create.krnl, &outerScope);
                SymIE l(loopInd[0]);
                getIndexExprList<SymIE>(memAF, inputAF);
                // E1: add the "l" local E1 offset.
                inputAF[E1] = inputAF[E1] + l;
                // Load the f32.
                Value vecF32H[unrollVL], vecF32L[unrollVL], vecF16[unrollVL];
                for (int64_t u = 0; u < unrollVL; ++u) {
                  LitIE iH(u * archVL), iL(u * archVL + archVL / 2);
                  vecF32H[u] = create.vec.loadIE(
                      vecF32Type, input, inputAF, {iH.getValue()});
                  vecF32L[u] = create.vec.loadIE(
                      vecF32Type, input, inputAF, {iL.getValue()});
                }
                if (saturation) {
                  // Get rid of too-high values.
                  for (int64_t u = 0; u < unrollVL; ++u) {
                    vecF32H[u] = create.math.min(vecF32H[u], vecDlf16Max);
                    vecF32L[u] = create.math.min(vecF32L[u], vecDlf16Max);
                  }
                  // Get rid of too-low values.
                  for (int64_t u = 0; u < unrollVL; ++u) {
                    vecF32H[u] = create.math.max(vecF32H[u], vecDlf16Min);
                    vecF32L[u] = create.math.max(vecF32L[u], vecDlf16Min);
                  }
                }
                // Convert f32 to dlfloat16.
                for (int64_t u = 0; u < unrollVL; ++u) {
                  vecF16[u] = rewriter.create<ZLowConvertF32ToDLF16VectorOp>(
                      loc, vecF32H[u], vecF32L[u]);
                }
                // Store the dlfloat16.
                for (int64_t u = 0; u < unrollVL; ++u) {
                  create.vec.storeIE(vecF16[u], allocAsTx64,
                      {SymIE(allocTileIndex), l + (u * archVL)}, {});
                }
              });
        });

    rewriter.eraseOp(stickOp);
    return success();
  }
};

/*!
 *  Function pass that optimizes ZLowIR.
 */
class ZLowStickExpansionPass
    : public PassWrapper<ZLowStickExpansionPass, OperationPass<func::FuncOp>> {

public:
  ZLowStickExpansionPass(bool enableParallel)
      : PassWrapper<ZLowStickExpansionPass, OperationPass<func::FuncOp>>(),
        enableParallel(enableParallel) {}

  bool enableParallel;

  StringRef getArgument() const override { return "zlow-stick-expansion"; }

  StringRef getDescription() const override {
    return "ZLow Stick/Unstick Ops expansion pass.";
  }

  void runOnOperation() override {
    Operation *function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<StickExpansionPattern>(&getContext(), enableParallel);
    patterns.insert<UnstickExpansionPattern>(&getContext(), enableParallel);
    // patterns.insert<UnstickExpansionPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> createZLowStickExpansionPass(bool enableParallel) {
  return std::make_unique<ZLowStickExpansionPass>(enableParallel);
}

} // namespace zlow
} // namespace onnx_mlir
