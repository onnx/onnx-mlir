/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- NonZero.cpp - Lowering NonZero Op ----------------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX NonZero Operator to Krnl dialect, as a flat-tiled
// stream compaction. See rewriteBlockCompaction.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXNonZeroOpLowering : public OpConversionPattern<ONNXNonZeroOp> {
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MathBuilder, MemRefBuilder, SCFBuilder, VectorBuilder>;

  // Number of blocks aimed for; the tile size is derived from it
  // (selectTileSize).
  static constexpr int64_t targetBlockNum = 64;
  // Tile size used when the shape is not fully static.
  static constexpr int64_t defaultTileSize = 1024;
  // Lower and upper bounds on the tile size search.
  static constexpr int64_t minTileSize = 64;
  static constexpr int64_t maxTileProbes = 4096;
  // Minimum block count for the block loops to be parallelized.
  static constexpr int64_t minBlocksForPar = 8;

  bool enableSIMD = false;
  bool enableParallel = false;

  ONNXNonZeroOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableSIMD, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx), enableSIMD(enableSIMD) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXNonZeroOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ONNXNonZeroOp nonZeroOp,
      ONNXNonZeroOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    return rewriteBlockCompaction(nonZeroOp, adaptor, rewriter);
  }

private:
  //===--------------------------------------------------------------------===//
  // New implementation: flat-tiled stream compaction.
  //===--------------------------------------------------------------------===//
  //
  // NonZero is a stream compaction: scan X in row-major order and, for every
  // nonzero element, append its coordinates to the output. The output has shape
  // [R, N] where N is a runtime value, so out[a][k] is the a-th coordinate of
  // the k-th nonzero.
  //
  // X is viewed as one flat run of M = D[0]*...*D[R-1] elements, cut into
  // blocks of `tileSize`. Flat index order *is* row-major order, so blocks are
  // visited in the order the ONNX spec requires and each block owns a
  // contiguous, increasing range of output columns.
  //
  // Blocks do not respect row boundaries: a block is tileSize elements of the
  // flat run, whatever rows those fall in.
  //
  //   pass 1 (parallel over blocks)
  //       nzPerBlock[b+1] = number of nonzeros in block b
  //   pass 2 (serial, NB+1 iterations)
  //       in-place inclusive scan. Because pass 1 wrote the counts shifted up
  //       by one, the result is the *exclusive* prefix sum, so afterwards
  //         nzPerBlock[b]   = first output column owned by block b
  //         nzPerBlock[b+1] = one past the last column owned by block b
  //         nzPerBlock[NB]  = N
  //   pass 3 (parallel over blocks)
  //       block b writes only columns [nzPerBlock[b], nzPerBlock[b+1]), which
  //       are disjoint across blocks, so no synchronization is needed.
  //
  // Coordinates are regenerated from the flat index by successive division (see
  // storeCoordinates), inside the "is it nonzero" test, so once per nonzero.
  //
  // Pass 1 is vectorized with krnl.simdReduceIE when the element type has
  // vector support. Pass 3 is still scalar.
  LogicalResult rewriteBlockCompaction(ONNXNonZeroOp nonZeroOp,
      ONNXNonZeroOpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    Operation *op = nonZeroOp.getOperation();
    Location loc = ONNXLoc<ONNXNonZeroOp>(op);
    IndexExprScope outerScope(&rewriter, loc);
    MDBuilder create(rewriter, loc);

    Value X = adaptor.getX();
    MemRefType xMemRefType = mlir::cast<MemRefType>(X.getType());
    int64_t xRank = xMemRefType.getRank();
    Type xElementType = xMemRefType.getElementType();

    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType resMemRefType = mlir::cast<MemRefType>(convertedType);
    Type resElementType = resMemRefType.getElementType();

    Type indexTy = rewriter.getIndexType();
    Type i1Ty = rewriter.getI1Type();
    MemRefType scalarIndexType = MemRefType::get({}, indexTy);
    Value iZero = create.math.constantIndex(0);
    Value iOne = create.math.constantIndex(1);
    Value xZero = create.math.constant(xElementType, 0);
    Value falseVal = create.math.constant(i1Ty, 0);
    Value trueVal = create.math.constant(i1Ty, 1);

    // Rank 0 is degenerate: the result has zero rows, so there are no
    // coordinates to write and only the dynamic dimension has to be computed.
    // Note numpy rejects nonzero() on a 0-d array outright, so there is no
    // reference semantics to match here beyond not crashing.
    if (xRank == 0) {
      Value x = create.krnl.load(X, {});
      Value n = create.math.select(create.math.eq(x, xZero), iZero, iOne);
      SmallVector<IndexExpr, 2> outDims0;
      outDims0.emplace_back(LitIE(0));
      outDims0.emplace_back(DimIE(n));
      Value res0 = create.mem.alignedAlloc(resMemRefType, outDims0);
      rewriter.replaceOp(op, res0);
      onnxToKrnlSimdReport(op, /*successful*/ false, /*vectorLength*/ 0,
          /*simdLoopTripCount*/ 0, "rank 0");
      return success();
    }

    // Input dims, as IndexExpr and as Values. The loop bodies below use the
    // Values: they are in nested regions, where an IndexExpr from this scope is
    // not usable.
    DimsExpr xDims;
    create.krnlIE.getShapeAsDims(X, xDims);
    SmallVector<Value, 4> xDimVals;
    for (int64_t a = 0; a < xRank; ++a)
      xDimVals.emplace_back(xDims[a].getValue());

    // Total element count.
    IndexExpr mIE = LitIE(1);
    for (int64_t a = 0; a < xRank; ++a)
      mIE = mIE * xDims[a];
    Value mVal = mIE.getValue();

    // SIMD for the counting pass. Vectorized when the element type has vector
    // support; i1 does not, so a bool input counts scalar. VL comes from the
    // input type rather than from the accumulator: this pass is memory bound,
    // so bytes per load is what matters, and a wider accumulator only costs
    // registers.
    int64_t VL = VectorMachineSupport::getArchVectorLength(
        GenericOps::ArithmeticGop, xElementType);
    bool doSimd = enableSIMD && VL > 1;
    if (!doSimd)
      VL = 1;

    // Tile size, and whether a short final block is possible. staticSize is M',
    // the product of the static dims only. M = M' * (product of the dynamic
    // dims), so a tile dividing M' divides M, and needTail is false.
    int64_t staticSize;
    bool allStatic =
        MemRefBuilder::getStaticMemSize(xMemRefType, staticSize, xRank);
    int64_t tileSize = selectTileSize(staticSize, allStatic, VL);
    bool needTail = (staticSize % tileSize) != 0;
    // With VL dividing the tile, a full block is entirely SIMD: simdReduceIE
    // emits no scalar remainder loop at all.
    bool fullySimd = tileSize % VL == 0;

    IndexExpr nbIE = mIE.ceilDiv(tileSize);
    IndexExpr nbPlus1IE = nbIE + 1;
    Value tVal = create.math.constantIndex(tileSize);

    // Report the tile size, the block count, and which constraint picked the
    // tile; none of that is visible in the generated IR.
    LLVM_DEBUG({
      const char *why;
      if (!allStatic)
        why = "dynamic shape, fixed default";
      else if (staticSize > 0 &&
               staticSize <=
                   std::max(minTileSize, static_cast<int64_t>(llvm::divideCeil(
                                             staticSize, targetBlockNum))))
        why = "input fits in one block";
      else if (llvm::divideCeil(staticSize, targetBlockNum) <
               (uint64_t)minTileSize)
        why = "min tile floor, so NB is below target";
      else
        why = "block-count target";
      llvm::dbgs() << "NonZero flat tiling: rank " << xRank << ", M' "
                   << staticSize << (allStatic ? " (exact)" : " (static part)")
                   << ", tileSize " << tileSize << " (" << why << ")"
                   << ", tail path " << needTail << ", NB "
                   << (nbIE.isLiteral() ? nbIE.getLiteral() : -1) << ", VL "
                   << VL
                   << (doSimd ? (fullySimd ? " (fully simd)"
                                           : " (simd + leftover)")
                              : " (no simd)")
                   << "\n";
    });

    // A flat 1-D view of X, walked by flat index by both scans. Rank 1 is
    // already flat and the helper returns X unchanged.
    //
    // Do not use memref.collapse_shape here: expanding one later emits an
    // affine.delinearize_index, which ConvertKrnlToLLVMPass fails to legalize.
    DimsExpr flatDims;
    Value xFlat = create.mem.reshapeToFlatInnermost(
        X, xDims, flatDims, /*flatten*/ xRank);

    // nzPerBlock[0..NB]. Generally dynamically sized, so heap allocated.
    int64_t nbPlus1Lit =
        nbPlus1IE.isLiteral() ? nbPlus1IE.getLiteral() : ShapedType::kDynamic;
    SmallVector<IndexExpr, 1> nzDims(1, nbPlus1IE);
    Value nzPerBlock =
        create.mem.alignedAlloc(MemRefType::get({nbPlus1Lit}, indexTy), nzDims);
    // Slots 1..NB are written unconditionally by pass 1; slot 0 is seeded by
    // pass 2. No initialization is needed here.

    bool doPar = enableParallel &&
                 (!nbIE.isLiteral() || nbIE.getLiteral() >= minBlocksForPar);
    onnxToKrnlParallelReport(op, doPar, /*loopLevel*/ 0, LitIE(0), nbIE,
        doPar ? "block loop" : "too few blocks");

    // Each pass keeps one running scalar: pass 1 a nonzero count, pass 2 the
    // scan total, pass 3 the output column. The three uses do not overlap in
    // time, so one alloca outside the loops serves all of them. Pass 2 always
    // uses it.
    //
    // When the block loops are parallel, passes 1 and 3 allocate inside the
    // block body instead: one alloca outside would be shared by every thread.
    // ProcessScfParallelPrivate wraps parallel bodies in memref.alloca_scope,
    // so an alloca there is thread private and reclaimed per iteration.
    Value seqTmp = create.mem.alloca(scalarIndexType);
    auto runningTmp = [&](const MDBuilder &create) {
      return doPar ? create.mem.alloca(scalarIndexType) : seqTmp;
    };

    //===------------------------------------------------------------------===//
    // Pass 1: count the nonzeros of every block.
    //===------------------------------------------------------------------===//
    // The accumulator is i64, not the input type: a count does not fit in an i1
    // or an i8 lane, and i64 matches the index type that nzPerBlock holds.
    Type accTy = rewriter.getIntegerType(64);
    Value accZero = create.math.constant(accTy, 0);
    Value accOne = create.math.constant(accTy, 1);
    // simdReduceIE keeps VL partial sums in a temp, reduced to a scalar at the
    // end. Allocated per block when parallel, once otherwise, as for seqTmp.
    MemRefType simdTmpType = MemRefType::get({VL}, accTy);
    // Assigned in an if, not a ternary: the ternary's common type would be
    // memref::AllocaOp, so the null branch would build a null op and converting
    // that to a Value dereferences it.
    Value seqSimdTmp;
    if (doSimd && !doPar)
      seqSimdTmp = create.mem.alignedAlloca(simdTmpType);

    onnxToKrnlSimdReport(op, doSimd, doSimd ? VL : 0, tileSize,
        doSimd ? "counting pass" : "no simd for this element type");

    create.krnl.forLoopIE(LitIE(0), nbIE, /*step*/ 1, doPar,
        [&](const KrnlBuilder &kb, ValueRange blockInd) {
          MDBuilder create(kb);
          IndexExprScope blockScope(create.krnl);
          Value b = blockInd[0];
          Value lo = create.math.mul(b, tVal);
          DimsExpr outputAF = {DimIE(b) + 1};

          // Count exactly tileSize elements from lo, in SIMD.
          auto countFullSimd = [&](const MDBuilder &create) {
            Value tmp =
                doPar ? create.mem.alignedAlloca(simdTmpType) : seqSimdTmp;
            DimsExpr inputAF = {DimIE(lo)}, tmpAF = {LitIE(0)};
            create.krnl.simdReduceIE(LitIE(0), LitIE(tileSize), VL, fullySimd,
                {xFlat}, {inputAF}, {tmp}, {tmpAF}, {nzPerBlock}, {outputAF},
                {accZero},
                {[&](const KrnlBuilder &kb, Value in, Value acc, int64_t vl) {
                  MathBuilder createMath(kb);
                  // eq against zero inverted by the select, not neq:
                  // MathBuilder::neq emits arith.cmpf ONE for floats, which is
                  // false for NaN, and ONNX counts NaN as nonzero.
                  Value inc = createMath.select(
                      createMath.eq(in, xZero), accZero, accOne);
                  return createMath.add(acc, inc);
                }},
                {[&](const KrnlBuilder &kb, Value acc, int64_t vl) {
                  MDBuilder create(kb);
                  Value sum = create.vec.reduction(
                      VectorBuilder::CombiningKind::ADD, acc);
                  return create.math.cast(indexTy, sum);
                }});
          };

          // Scalar count, bound checked or not, into the running scalar.
          auto countScalar = [&](const MDBuilder &create, bool guarded) {
            Value cnt = runningTmp(create);
            create.krnl.store(iZero, cnt);
            emitScan(create, lo, tVal, mVal, guarded,
                [&](const MDBuilder &create, Value m) {
                  Value x = create.krnl.load(xFlat, {m});
                  Value inc =
                      create.math.select(create.math.eq(x, xZero), iZero, iOne);
                  create.krnl.store(
                      create.math.add(create.krnl.load(cnt), inc), cnt);
                });
            create.krnl.store(
                create.krnl.load(cnt), nzPerBlock, {create.math.add(b, iOne)});
          };

          auto countFull = [&](const MDBuilder &create) {
            if (doSimd)
              countFullSimd(create);
            else
              countScalar(create, /*guarded*/ false);
          };

          if (!needTail) {
            countFull(create);
            return;
          }
          // A short final block cannot be vectorized over the full tile, so it
          // takes the bound-checked scalar path.
          create.scf.ifThenElse(
              create.math.sle(create.math.add(lo, tVal), mVal),
              [&](const SCFBuilder &sb) {
                MDBuilder c(sb);
                countFull(c);
              },
              [&](const SCFBuilder &sb) {
                MDBuilder c(sb);
                countScalar(c, /*guarded*/ true);
              });
        });

    //===------------------------------------------------------------------===//
    // Pass 2: exclusive prefix sum, then allocate the output.
    //===------------------------------------------------------------------===//
    // Serial, NB+1 iterations, i.e. one per tileSize input elements.
    //
    // Slot 0 is seeded here. With the counts shifted up by one, an inclusive
    // scan from a zero at slot 0 gives the exclusive prefix sum. Pass 3 reads
    // slot 0 as block 0's first output column; when NB == 0 it is the only slot
    // read.
    create.krnl.store(iZero, nzPerBlock, {iZero});
    Value runMem = seqTmp;
    create.krnl.store(iZero, runMem);
    create.krnl.forLoopIE(LitIE(0), nbPlus1IE, /*step*/ 1, /*parallel*/ false,
        [&](const KrnlBuilder &kb, ValueRange scanInd) {
          MathBuilder createMath(kb);
          Value b = scanInd[0];
          Value run = createMath.add(kb.load(runMem), kb.load(nzPerBlock, {b}));
          kb.store(run, runMem);
          kb.store(run, nzPerBlock, {b});
        });
    // After the scan runMem holds nzPerBlock[NB], the total nonzero count.
    Value n = create.krnl.load(runMem);

    SmallVector<IndexExpr, 2> outDims;
    outDims.emplace_back(LitIE(xRank));
    outDims.emplace_back(DimIE(n));
    Value resMemRef = create.mem.alignedAlloc(resMemRefType, outDims);

    //===------------------------------------------------------------------===//
    // Pass 3: write the coordinates.
    //===------------------------------------------------------------------===//
    create.krnl.forLoopIE(LitIE(0), nbIE, /*step*/ 1, doPar,
        [&](const KrnlBuilder &kb, ValueRange blockInd) {
          MDBuilder create(kb);
          Value b = blockInd[0];
          Value kStart = create.krnl.load(nzPerBlock, {b});
          Value kEnd = create.krnl.load(nzPerBlock, {create.math.add(b, iOne)});
          Value kMem = runningTmp(create);
          // Skip blocks that pass 1 found empty, so their data is not
          // rescanned.
          create.scf.ifThenElse(
              create.math.slt(kStart, kEnd), [&](const SCFBuilder &createSCF) {
                MDBuilder create(createSCF);
                Value lo = create.math.mul(b, tVal);
                create.krnl.store(kStart, kMem);
                emitBlockScan(create, lo, tVal, mVal, needTail,
                    [&](const MDBuilder &create, Value m) {
                      Value x = create.krnl.load(xFlat, {m});
                      // eq inverted by the select, as in pass 1.
                      Value isNonZero = create.math.select(
                          create.math.eq(x, xZero), falseVal, trueVal);
                      create.scf.ifThenElse(
                          isNonZero, [&](const SCFBuilder &b2) {
                            MDBuilder c(b2);
                            Value k = c.krnl.load(kMem);
                            storeCoordinates(c, m, xDimVals, xRank,
                                resElementType, resMemRef, k);
                            c.krnl.store(c.math.add(k, iOne), kMem);
                          });
                    });
              });
        });

    rewriter.replaceOp(op, resMemRef);
    return success();
  }

  // Pick the elements-per-block: aim for targetBlockNum blocks, then take a
  // divisor of staticSize (M') near the tile size that implies, searching
  // outward in both directions. A divisor of M' also divides M, so needTail is
  // then false.
  //
  // Among those, a tile that is also a multiple of VL is preferred, because
  // then a full block is entirely SIMD and simdReduceIE emits no scalar
  // remainder. Such a tile exists only when VL divides M', so this is a
  // preference and not a requirement -- and the right way round of the two:
  // giving up divisibility by M' would leave one short block of up to T
  // elements scalar (~1/targetBlockNum of the input), whereas giving up
  // divisibility by VL leaves only (T mod VL) elements per block, which is
  // smaller by orders of magnitude.
  static int64_t selectTileSize(
      int64_t staticSize, bool allStatic, int64_t VL) {
    // Not fully static: M' is unrelated to the real element count, so use the
    // fixed tile as the target. The divisor search below still applies.
    int64_t target = defaultTileSize;
    if (allStatic) {
      int64_t perBlock =
          static_cast<int64_t>(llvm::divideCeil(staticSize, targetBlockNum));
      target = std::max(minTileSize, perBlock);
    }
    // A fully static input no bigger than one tile is exactly one block.
    if (allStatic && staticSize > 0 && staticSize <= target)
      return staticSize;
    // A tile larger than M' cannot divide it. First pass insists on a multiple
    // of VL, second pass drops that.
    if (staticSize > 0)
      for (int64_t vlStep : {VL, (int64_t)1})
        for (int64_t d = 0; d < maxTileProbes; ++d)
          for (int64_t t : {target - d, target + d}) {
            if (t < minTileSize || t > staticSize)
              continue;
            if (t % vlStep == 0 && staticSize % t == 0)
              return t;
            if (d == 0)
              break; // target probed once, not twice.
          }
    // No divisor found: use the target, and needTail will be true.
    return target;
  }

  // One scalar scan of the tileSize elements starting at flat index lo,
  // invoking bodyFn on each. The trip count is the constant tileSize either
  // way; when guarded, each element is bound checked against mVal, which is
  // what makes the same constant-bounded loop usable for a short final block.
  static void emitScan(const MDBuilder &create, Value lo, Value tVal,
      Value mVal, bool guarded,
      function_ref<void(const MDBuilder &, Value)> bodyFn) {
    Value zero = create.math.constantIndex(0);
    ValueRange oLoop = create.krnl.defineLoops(1);
    create.krnl.iterate(oLoop, oLoop, {zero}, {tVal},
        [&](const KrnlBuilder &ck, ValueRange oInd) {
          MDBuilder c(ck);
          Value m = c.math.add(lo, oInd[0]);
          if (!guarded) {
            bodyFn(c, m);
            return;
          }
          c.scf.ifThenElse(c.math.slt(m, mVal), [&](const SCFBuilder &b) {
            MDBuilder c2(b);
            bodyFn(c2, m);
          });
        });
  }

  // emitScan, plus a per-block test selecting the bound-checked variant for a
  // short final block when one is possible at all.
  static void emitBlockScan(const MDBuilder &create, Value lo, Value tVal,
      Value mVal, bool needTail,
      function_ref<void(const MDBuilder &, Value)> bodyFn) {
    if (!needTail) {
      emitScan(create, lo, tVal, mVal, /*guarded*/ false, bodyFn);
      return;
    }
    create.scf.ifThenElse(
        create.math.sle(create.math.add(lo, tVal), mVal),
        [&](const SCFBuilder &b) {
          MDBuilder c(b);
          emitScan(c, lo, tVal, mVal, /*guarded*/ false, bodyFn);
        },
        [&](const SCFBuilder &b) {
          MDBuilder c(b);
          emitScan(c, lo, tVal, mVal, /*guarded*/ true, bodyFn);
        });
  }

  // Regenerate the R coordinates of flat index m and store them into column k,
  // one per output row.
  //
  // Successive division from the innermost axis outward, R-1 divisions in
  // total. The remainder is t - (t / D)*D rather than a separate modulo, so
  // each level costs one division. At rank 1 the loop does not run and the flat
  // index is the coordinate.
  //
  // Do not use affine.delinearize_index for this: ConvertKrnlToLLVMPass fails
  // to legalize it, as the AffineToStd pattern set it installs does not cover
  // it.
  static void storeCoordinates(const MDBuilder &create, Value m,
      ArrayRef<Value> xDimVals, int64_t xRank, Type resElementType,
      Value resMemRef, Value k) {
    Value t = m;
    for (int64_t a = xRank - 1; a >= 1; --a) {
      Value aVal = create.math.constantIndex(a);
      Value quot = create.math.floorDiv(t, xDimVals[a]);
      Value rem = create.math.sub(t, create.math.mul(quot, xDimVals[a]));
      create.krnl.store(
          create.math.cast(resElementType, rem), resMemRef, {aVal, k});
      t = quot;
    }
    // The outermost coordinate is the remaining quotient.
    Value zero = create.math.constantIndex(0);
    create.krnl.store(
        create.math.cast(resElementType, t), resMemRef, {zero, k});
  }
};

void populateLoweringONNXNonZeroOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD,
    bool enableParallel) {
  patterns.insert<ONNXNonZeroOpLowering>(
      typeConverter, ctx, enableSIMD, enableParallel);
}

} // namespace onnx_mlir
