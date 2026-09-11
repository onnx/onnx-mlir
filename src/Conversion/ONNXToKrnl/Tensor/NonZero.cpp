/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- NonZero.cpp - Lowering NonZero Op ----------------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX NonZero Operator to Krnl dialect.
//
// Two implementations coexist while the new one is being validated; select with
// --test-compiler-opt. See matchAndRewrite.
//
//===----------------------------------------------------------------------===//

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXNonZeroOpLowering : public OpConversionPattern<ONNXNonZeroOp> {
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MathBuilder, MemRefBuilder, SCFBuilder>;

  // How many blocks to aim for. The tile size is derived from this, not the other
  // way round: the parallel loop becomes an omp.wsloop with no schedule clause,
  // i.e. static, so every thread is handed one contiguous chunk decided up front
  // and *extra blocks do nothing for load balance* -- clustered nonzeros land on
  // one thread whether there are 64 blocks or 3000. Past a small multiple of the
  // thread count, more blocks only lengthen the serial prefix sum of pass 2 and
  // grow the count array. If the schedule ever becomes dynamic or guided,
  // over-decomposition starts to pay and this should go up.
  static constexpr int64_t targetBlockNum = 64; // ~4x a practical 16 threads.
  // Never go below this many elements per block: the per-block overhead (a count
  // slot, the empty-block test, the loop setup) has to amortize over something.
  static constexpr int64_t minTileSize = 256;
  // How far the divisor search may probe. Bounds compile time when M' has no
  // divisor anywhere near the target, e.g. when it is a large prime.
  static constexpr int64_t maxTileProbes = 4096;
  // Below this many blocks, parallelizing is not worth the fork/join.
  static constexpr int64_t minBlocksForPar = 8;

  bool enableParallel = false;

  ONNXNonZeroOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXNonZeroOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ONNXNonZeroOp nonZeroOp,
      ONNXNonZeroOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Two lowerings are available:
    //
    // - rewriteMarginalSum: the historical implementation. It is INCORRECT for
    //   inputs of rank >= 2 (see the comment on that function) but correct for
    //   rank 1, which makes it a usable reference oracle for rank-1 inputs,
    //   e.g. via utils/CheckONNXModel.py. Kept for exactly that reason.
    // - rewriteBlockCompaction: the replacement. Correct for every rank.
    //
    // Once the new path is validated, flip the default, delete the old function
    // and stop consulting the flag.
    Value X = adaptor.getX();
    int64_t xRank = mlir::cast<MemRefType>(X.getType()).getRank();
    if (debugTestCompilerOpt) {
      // Rank 0 gives a degenerate 0 x N result; leave it on the old path rather
      // than carrying a special case through the new one.
      if (xRank > 0)
        return rewriteBlockCompaction(nonZeroOp, adaptor, rewriter);
      LLVM_DEBUG(llvm::dbgs() << "NonZero: rank 0, using the old lowering\n");
    }
    return rewriteMarginalSum(nonZeroOp, adaptor, rewriter);
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
  // X is viewed as one flat run of M = D[0]*...*D[R-1] elements, cut into blocks
  // of `tileSize`. Flat index order *is* row-major order, so blocks are visited
  // in the order the ONNX spec requires and each block owns a contiguous,
  // increasing range of output columns.
  //
  // Blocks deliberately do NOT respect row boundaries. Nothing in the algorithm
  // needs them to, and a fixed block size regardless of shape is what avoids
  // degenerating to one block per row when the innermost dimension is short: for
  // 1000x1000x3, row-aligned blocking would give 10^6 blocks and make pass 2's
  // serial scan proportional to the input.
  //
  //   pass 1 (parallel over blocks)
  //       nzPerBlock[b+1] = number of nonzeros in block b
  //   pass 2 (serial, NB+1 iterations)
  //       in-place inclusive scan. Because pass 1 wrote the counts shifted up by
  //       one, the result is the *exclusive* prefix sum, so afterwards
  //         nzPerBlock[b]   = first output column owned by block b
  //         nzPerBlock[b+1] = one past the last column owned by block b
  //         nzPerBlock[NB]  = N
  //   pass 3 (parallel over blocks)
  //       block b writes only columns [nzPerBlock[b], nzPerBlock[b+1]). Those
  //       ranges are disjoint by construction, so no two blocks ever touch the
  //       same output element and no synchronization is needed.
  //
  // Since a block spans arbitrary rows, a nonzero's coordinates are not loop
  // indices. They are regenerated from the flat index by successive division
  // (see storeCoordinates), emitted *inside* the "is it nonzero" test so that it
  // runs per nonzero rather than per element.
  //
  // Deliberately NOT done yet, to keep this reviewable:
  //  - SIMD. Pass 1's full path is a contiguous masked reduce-sum with a
  //    constant trip count and no bound check, which is exactly the shape
  //    krnl.simdReduceIE wants. Pass 3's scan should gate each vector on "does
  //    any lane hold a nonzero" before entering the scalar per-lane code.
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
    assert(xRank > 0 && "expected a ranked, non-scalar input");

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

    // Input dims, as IndexExpr and as Values: the loop bodies below live in
    // nested regions, where plain Values are simpler to reason about than
    // IndexExpr (which is tied to the scope it was built in).
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

    // Tile size, and whether a short final block is possible at all. M' is the
    // product of the *static* dims only. Because M = M' * (product of the
    // dynamic dims), a tile size dividing M' divides M too, whatever the dynamic
    // dims turn out to be at run time -- so picking a divisor of M' proves here
    // that every block is full, even for a dynamically shaped input.
    int64_t staticSize;
    bool allStatic =
        MemRefBuilder::getStaticMemSize(xMemRefType, staticSize, xRank);
    int64_t tileSize = selectTileSize(staticSize, allStatic);
    bool needTail = (staticSize % tileSize) != 0;

    IndexExpr nbIE = mIE.ceilDiv(tileSize);
    IndexExpr nbPlus1IE = nbIE + 1;
    Value tVal = create.math.constantIndex(tileSize);

    LLVM_DEBUG(llvm::dbgs()
               << "NonZero flat tiling: rank " << xRank << ", M' " << staticSize
               << (allStatic ? " (exact)" : " (static part only)")
               << ", tileSize " << tileSize << ", tail path " << needTail
               << ", NB " << (nbIE.isLiteral() ? nbIE.getLiteral() : -1) << "\n");

    // A flat 1-D view of X, so that both scans can walk it by flat index without
    // reconstructing coordinates. Rank 1 already is flat, and the helper returns
    // the value unchanged in that case.
    //
    // This deliberately does NOT use memref.collapse_shape, even though that
    // would be pure metadata: expanding a collapse_shape later in the pipeline
    // emits an affine.delinearize_index to map the flat index back onto the
    // original dimensions, and ConvertKrnlToLLVMPass then fails to legalize that
    // op. reshapeToFlatInnermost costs a small buffer holding the target shape
    // (once per NonZero op, not per element) and is what the other flattening
    // lowerings here use -- Math/Elementwise.cpp, Math/Reduction.cpp,
    // NN/Normalization.cpp -- so it is known to survive to LLVM.
    DimsExpr flatDims;
    Value xFlat =
        create.mem.reshapeToFlatInnermost(X, xDims, flatDims, /*flatten*/ xRank);

    // nzPerBlock[0..NB]. Generally dynamically sized, so heap allocated.
    int64_t nbPlus1Lit =
        nbPlus1IE.isLiteral() ? nbPlus1IE.getLiteral() : ShapedType::kDynamic;
    SmallVector<IndexExpr, 1> nzDims(1, nbPlus1IE);
    Value nzPerBlock = create.mem.alignedAlloc(
        MemRefType::get({nbPlus1Lit}, indexTy), nzDims);
    // Slot 0 stays zero; pass 1 fills slots 1..NB.
    create.krnl.store(iZero, nzPerBlock, {iZero});

    bool doPar = enableParallel &&
                 (!nbIE.isLiteral() || nbIE.getLiteral() >= minBlocksForPar);
    onnxToKrnlParallelReport(op, doPar, /*loopLevel*/ 0, LitIE(0), nbIE,
        doPar ? "block loop" : "too few blocks");

    //===------------------------------------------------------------------===//
    // Pass 1: count the nonzeros of every block.
    //===------------------------------------------------------------------===//
    create.krnl.forLoopIE(LitIE(0), nbIE, /*step*/ 1, doPar,
        [&](const KrnlBuilder &kb, ValueRange blockInd) {
          MDBuilder create(kb);
          Value b = blockInd[0];
          Value lo = create.math.mul(b, tVal);
          // Scalar accumulator, declared inside the block body so that it is
          // thread private when this loop runs in parallel.
          Value cnt = create.mem.alloca(scalarIndexType);
          create.krnl.store(iZero, cnt);
          emitBlockScan(create, lo, tVal, mVal, needTail,
              [&](const MDBuilder &create, Value m) {
                Value x = create.krnl.load(xFlat, {m});
                Value inc = create.math.select(
                    create.math.eq(x, xZero), iZero, iOne);
                create.krnl.store(
                    create.math.add(create.krnl.load(cnt), inc), cnt);
              });
          create.krnl.store(create.krnl.load(cnt), nzPerBlock,
              {create.math.add(b, iOne)});
        });

    //===------------------------------------------------------------------===//
    // Pass 2: exclusive prefix sum, then allocate the output.
    //===------------------------------------------------------------------===//
    // NB+1 iterations, i.e. one per tileSize input elements, so leaving this
    // serial costs about 0.1% of the work at the default tile size.
    Value runMem = create.mem.alloca(scalarIndexType);
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
          Value kEnd =
              create.krnl.load(nzPerBlock, {create.math.add(b, iOne)});
          // Skip empty blocks outright. Pass 1 already established there is
          // nothing here, so rescanning the data would be pure waste. This is a
          // block-granularity version of the per-vector "any lane nonzero" gate
          // that the SIMD version will add.
          create.scf.ifThenElse(create.math.slt(kStart, kEnd),
              [&](const SCFBuilder &createSCF) {
                MDBuilder create(createSCF);
                Value lo = create.math.mul(b, tVal);
                // The store is guarded rather than unconditional-and-advancing:
                // the branchless form writes one element past the block's range
                // whenever the block's trailing elements are zero, and that
                // element belongs to the next block, i.e. to another thread.
                Value kMem = create.mem.alloca(scalarIndexType);
                create.krnl.store(kStart, kMem);
                emitBlockScan(create, lo, tVal, mVal, needTail,
                    [&](const MDBuilder &create, Value m) {
                      Value x = create.krnl.load(xFlat, {m});
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
    onnxToKrnlSimdReport(op, /*successful*/ false, /*vectorLength*/ 0,
        /*simdLoopTripCount*/ 0, "no simd yet in flat tiling");
    return success();
  }

  // Pick the elements-per-block. See the M' argument at the call site: any tile
  // size that divides the static element count also divides the true element
  // count, whatever the dynamic dims turn out to be, so a divisor of M' proves at
  // compile time that no block is short and lets the guarded tail path be dropped
  // entirely.
  //
  // Aim for targetBlockNum blocks and look for a divisor of M' near the tile size
  // that implies. The search probes *outward* in both directions, which matters:
  // probing only downward loses tail-freeness for mid-size inputs, e.g. M' = 3600
  // would stop at 256 (which does not divide it) and miss the 240 just below.
  static int64_t selectTileSize(int64_t staticSize, bool allStatic) {
    int64_t perBlock =
        static_cast<int64_t>(llvm::divideCeil(staticSize, targetBlockNum));
    int64_t target = std::max(minTileSize, perBlock);
    // A fully static input no bigger than one tile is exactly one block.
    if (allStatic && staticSize > 0 && staticSize <= target)
      return staticSize;
    // A tile larger than M' cannot divide it, so the search is bounded above by
    // M' as well as by the probe budget.
    if (staticSize > 0)
      for (int64_t d = 0; d < maxTileProbes; ++d)
        for (int64_t t : {target - d, target + d}) {
          if (t < minTileSize || t > staticSize)
            continue;
          if (staticSize % t == 0)
            return t;
          if (d == 0)
            break; // target probed once, not twice.
        }
    // No usable divisor: keep the target and pay for the tail path instead.
    return target;
  }

  // Scan the tileSize elements starting at flat index lo, invoking bodyFn on
  // each. The full path has a constant trip count and no bound check; a short
  // final block, when one is possible at all, takes a guarded path. That path
  // runs at most once per tensor, so the guard costs nothing measurable, and
  // keeping both trip counts constant keeps the loops affine (a bound derived
  // from the enclosing loop index is not).
  static void emitBlockScan(const MDBuilder &create, Value lo, Value tVal,
      Value mVal, bool needTail,
      function_ref<void(const MDBuilder &, Value)> bodyFn) {
    auto emitLoop = [&](const MDBuilder &create, bool guarded) {
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
    };
    if (!needTail) {
      emitLoop(create, /*guarded*/ false);
      return;
    }
    Value isFull = create.math.sle(create.math.add(lo, tVal), mVal);
    create.scf.ifThenElse(
        isFull,
        [&](const SCFBuilder &b) {
          MDBuilder c(b);
          emitLoop(c, /*guarded*/ false);
        },
        [&](const SCFBuilder &b) {
          MDBuilder c(b);
          emitLoop(c, /*guarded*/ true);
        });
  }

  // Regenerate the R coordinates of flat index m and store them into column k,
  // one per output row.
  //
  // Successive division from the innermost axis outward. The remainder is formed
  // as t - (t / D)*D rather than with a separate modulo, which makes it explicit
  // that each level costs ONE division: R-1 in total, which is the floor for R
  // independent coordinates. With a compile-time constant dim each becomes a
  // multiply-high and a shift, with no division instruction at all. Rank 1 needs
  // no special case: the loop simply does not run and the flat index is the
  // coordinate.
  //
  // affine.delinearize_index would express this as a single op and would be the
  // nicer form, but it does not survive this pipeline: ConvertKrnlToLLVMPass
  // fails to legalize it, since the AffineToStd pattern set it installs
  // (ConvertKrnlToLLVM.cpp) does not cover that op -- even though --lower-affine
  // handles it in isolation. Little is lost here: these values are stored data,
  // not loop bounds or memref subscripts, so nothing downstream needs to reason
  // about them affinely. Revisit if that pass ever gains the pattern.
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
    // The outermost coordinate is whatever is left; no division needed.
    Value zero = create.math.constantIndex(0);
    create.krnl.store(create.math.cast(resElementType, t), resMemRef, {zero, k});
  }
  //===--------------------------------------------------------------------===//
  // Old implementation, kept as a reference oracle for rank-1 inputs.
  //===--------------------------------------------------------------------===//
  //
  // WARNING: this lowering is INCORRECT for inputs of rank >= 2. It computes,
  // for each axis, the marginal sum of the 0/1 mask over all *other* axes, then
  // reconstructs that axis's coordinates independently. Marginals do not
  // determine the joint index set, so any two inputs with equal marginals but
  // different nonzero patterns produce the same (hence wrong) answer. For
  // example these two have identical row and column sums:
  //
  //   [[0, 1],        gives (0,0),(1,1)   -- WRONG, should be (0,1),(1,0)
  //    [1, 0]]
  //
  //   [[1, 0],        gives (0,0),(1,1)   -- correct for this one
  //    [0, 1]]
  //
  // It is correct for rank 1, where the single marginal *is* the mask, and that
  // is the only reason it is still here: rank-1 inputs are common in real
  // models, so it serves as an independent oracle for the new lowering, e.g.
  //   utils/CheckONNXModel.py -m m.onnx -a "--test-compiler-opt=true"
  // Delete this function once the new path is validated.
  //
  /// Given an input of shape (3, 2):
  /// [[2, 1],
  /// [0, 2],
  /// [0, 1]]
  ///
  /// Output will be: [[0, 0, 1, 2], [0, 1, 1, 1]]
  /// The output's shape is (2, 4) where 2 is the input's rank, 4 is the number
  /// of nonzero values in the input.
  ///
  /// Step 1: Compute a 0-1 matrix:
  /// [[1, 1],
  /// [0, 1],
  /// [0, 1]]
  ///
  /// Step 2: Compute reduction sum for each dimension:
  /// rsum0 = ReduceSum(axis = 1) = [2, 1, 1]
  /// rsum1 = ReduceSum(axis = 0) = [1, 3]
  ///
  /// Step 3: Compute the number of nonzero for allocating the output buffer.
  ///
  /// Step 4: Compute output for each dimension, e.g. for dimension 0:
  /// ```
  ///   k = 0
  ///   for i range(len(rsum0)):
  ///     d = rsum0[i]
  ///     for j in range(d):
  ///       out[0][k+j] = i
  ///     k += d
  /// ```
  ///
  /// Note: in the following implementation:
  /// - Step1, Step2, and Step 3 are done with a single nested loop so the 0-1
  /// matrix is not generated explicitly.
  ///
  /// - Computation in Step 4 is optimized for trip count, but invalid when
  /// using 'affine.for'. 'affine.for' does not allow using 'affine.for'
  /// operands as bounds in another 'affine.for'. More info:
  /// llvm-project/mlir/test/Dialect/Affine/invalid.mlir
  ///
  /// Thus, We rewrite the loop in Step 4 into an affine-compatible one as
  /// follows:
  /// ```
  /// for i in range(nonzeroCount):
  ///   p = -1, s = 0
  ///   for j in range(len(rsum0)):
  ///      s += rsum0[j]
  ///      p = (i < s and p == -1) ? j : p
  ///   out[0][i] = p
  /// ```
  LogicalResult rewriteMarginalSum(ONNXNonZeroOp noneZeroOp,
      ONNXNonZeroOpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    Operation *op = noneZeroOp.getOperation();
    Location loc = ONNXLoc<ONNXNonZeroOp>(op);

    // Builder helper.
    IndexExprScope outerScope(&rewriter, loc);
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Frequently used MemRefType.
    Value X = adaptor.getX();
    MemRefType xMemRefType = mlir::cast<MemRefType>(X.getType());
    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType resMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t xRank = xMemRefType.getRank();

    // Frequently used element types.
    Type indexTy = rewriter.getIndexType();
    Type xElementType = xMemRefType.getElementType();
    Type resElementType = resMemRefType.getElementType();

    // Constant values.
    Value iZero = create.math.constantIndex(0);
    Value iOne = create.math.constantIndex(1);
    Value iMinusOne = create.math.constantIndex(-1);
    Value zero = create.math.constant(xElementType, 0);

    // Bounds for the input tensor.
    SmallVector<IndexExpr, 4> xLbs(xRank, LitIE(0));
    SmallVector<IndexExpr, 4> xUbs;
    create.krnlIE.getShapeAsDims(X, xUbs);

    // Emit a variable for the total number of nonzero values.
    // Scalar, ok to use alloca.
    Value nonzeroCount = create.mem.alloca(MemRefType::get({}, indexTy));
    create.krnl.store(iZero, nonzeroCount);

    // Emit alloc and dealloc for reduction sum along each dimension.
    // MemRefType: [Dxi64] where D is the dimension size.
    SmallVector<Value, 4> rsumMemRefs;
    for (int i = 0; i < xRank; ++i) {
      // Alloc and dealloc.
      IndexExpr xBound = create.krnlIE.getShapeAsDim(X, i);
      SmallVector<IndexExpr, 1> dimIE(1, xBound);
      int64_t dim =
          dimIE[0].isLiteral() ? dimIE[0].getLiteral() : ShapedType::kDynamic;
      Value alloc =
          create.mem.alignedAlloc(MemRefType::get({dim}, indexTy), dimIE);
      // Initialize to zero.
      ValueRange initLoopDef = create.krnl.defineLoops(1);
      create.krnl.iterate(initLoopDef, initLoopDef, {iZero},
          {xBound.getValue()},
          [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
            createKrnl.store(iZero, alloc, loopInd);
          });
      rsumMemRefs.emplace_back(alloc);
    }

    // Emit a loop for counting the total number of nonzero values, and
    // the reduction sum for each dimension.
    ValueRange rsumLoopDef = create.krnl.defineLoops(xMemRefType.getRank());
    create.krnl.iterateIE(rsumLoopDef, rsumLoopDef, xLbs, xUbs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          MathBuilder createMath(createKrnl);
          Value x = createKrnl.load(X, loopInd);
          Value eqCond = createMath.eq(x, zero);
          Value zeroOrOne = createMath.select(eqCond, iZero, iOne);
          // Count the total number of nonzero values.
          Value total = createKrnl.load(nonzeroCount);
          total = createMath.add(total, zeroOrOne);
          createKrnl.store(total, nonzeroCount);
          // Reduction sum of the number of nonzero values for each dimension.
          for (int64_t i = 0; i < xRank; ++i) {
            Value sum = createKrnl.load(rsumMemRefs[i], loopInd[i]);
            sum = createMath.add(sum, zeroOrOne);
            createKrnl.store(sum, rsumMemRefs[i], loopInd[i]);
          }
        });

    // Emit alloc and dealloc for the result of this operation.
    // MemRefType : [RxNxi64] where R is the input's rank, N is the number of
    // non zero values.
    Value numberOfZeros = create.krnl.load(nonzeroCount);
    SmallVector<IndexExpr, 2> dimExprs;
    dimExprs.emplace_back(LitIE(xRank));
    dimExprs.emplace_back(DimIE(numberOfZeros));
    Value resMemRef = create.mem.alignedAlloc(resMemRefType, dimExprs);

    // Emit code to compute the output for each dimension.
    // ```
    // for i in range(nonzeroCount):
    //   p = -1, s = 0
    //   for j in range(len(rsum0)):
    //      s += rsum0[j]
    //      p = (i < s and p == -1) ? j : p
    //   out[0][i] = p
    // ```

    // Scalars, ok to use alloca.
    Value pos = create.mem.alloca(MemRefType::get({}, indexTy));
    Value sum = create.mem.alloca(MemRefType::get({}, indexTy));
    ValueRange iLoopDef = create.krnl.defineLoops(1);
    create.krnl.iterate(iLoopDef, iLoopDef, {iZero}, {numberOfZeros},
        [&](const KrnlBuilder &ck, ValueRange iLoopInd) {
          MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
              MemRefBuilder>
              create(ck);
          Value i(iLoopInd[0]);
          for (int64_t axis = 0; axis < xRank; ++axis) {
            Value axisVal = create.math.constantIndex(axis);
            Value rsumBoundsVal = rsumMemRefs[axis];
            IndexExpr rsumBounds0 =
                create.krnlIE.getShapeAsDim(rsumBoundsVal, 0);

            create.krnl.store(iMinusOne, pos);
            create.krnl.store(iZero, sum);

            ValueRange jLoopDef = create.krnl.defineLoops(1);
            create.krnl.iterate(jLoopDef, jLoopDef, {iZero},
                {rsumBounds0.getValue()},
                [&](const KrnlBuilder &createKrnl, ValueRange jLoopInd) {
                  MathBuilder createMath(createKrnl);
                  Value j(jLoopInd[0]);
                  Value o = createKrnl.load(rsumMemRefs[axis], {j});
                  Value s = createKrnl.load(sum);
                  Value p = createKrnl.load(pos);
                  s = createMath.add(s, o);
                  Value andCond = createMath.andi(
                      createMath.slt(i, s), createMath.eq(p, iMinusOne));
                  p = createMath.select(andCond, j, p);
                  createKrnl.store(p, pos);
                  createKrnl.store(s, sum);
                });
            Value p = create.krnl.load(pos);
            p = create.math.cast(resElementType, p);
            create.krnl.store(p, resMemRef, {axisVal, i});
          }
        });

    rewriter.replaceOp(op, resMemRef);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXNonZeroOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXNonZeroOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
