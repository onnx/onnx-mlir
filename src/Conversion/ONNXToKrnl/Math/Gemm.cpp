/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Gemm.cpp - Lowering Gemm Op ------------------------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gemm Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

// Used to trace which op are used, good for profiling apps.
#define DEBUG_TYPE "gemm"
#define DEBUG_SIMD_OFF 0
#define DEBUG_UNROLL_OFF 0
#define DEBUG_OPTIMIZED_OFF 0

static constexpr int BUFFER_ALIGN = 128;

using namespace mlir;

namespace onnx_mlir {

template <typename GemmOp>
struct ONNXGemmOpLowering : public OpConversionPattern<GemmOp> {
  ONNXGemmOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableTiling, bool enableSIMD, bool enableParallel)
      : OpConversionPattern<GemmOp>(typeConverter, ctx),
        enableTiling(enableTiling), enableSIMD(enableSIMD) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXGemmOp::getOperationName());
  }

  using OpAdaptor = typename GemmOp::Adaptor;
  bool enableTiling;
  bool enableSIMD;
  bool enableParallel;

  void genericGemm(Operation *op, ONNXGemmOpAdaptor &adaptor, Type elementType,
      ONNXGemmOpShapeHelper &shapeHelper, Value alloc, Value zeroVal,
      Value alphaVal, Value betaVal, ConversionPatternRewriter &rewriter,
      Location loc, bool enableParallel) const {
    onnxToKrnlSimdReport(op, /*successful*/ false, /*vl*/ 0, /*trip count*/ 0,
        "no simd because tiling is disabled");

    // R is result (alloc).
    Value A(adaptor.getA()), B(adaptor.getB()), R(alloc);

    // Create all the loops at once (outer loops followed by inner loop).
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);
    ValueRange loopDef = create.krnl.defineLoops(3);
    SmallVector<Value, 2> outerLoopDef{loopDef[0], loopDef[1]};
    SmallVector<Value, 1> innerLoopDef{loopDef[2]};
    SmallVector<IndexExpr, 3> loopLbs(3, LitIE(0));
    IndexExpr outerUb0 = shapeHelper.getOutputDims()[0];
    IndexExpr outerUb1 = shapeHelper.getOutputDims()[1];
    IndexExpr innerUb = shapeHelper.aDims[1];
    SmallVector<IndexExpr, 3> loopUbs{outerUb0, outerUb1, innerUb};
    // Outer loops.
    auto plan = KrnlParallelPlan::noCollapse(outerLoopDef, /*first*/ 0,
        /*last excl*/ 1, {.minTripCount = 4});
    if (enableParallel)
      plan.tryCreateParallel(
          create.krnl, op, "generic GEMM on outer loop", loopLbs, loopUbs);
    create.krnl.iterateIE(loopDef, plan.optimizedLoopDef(), loopLbs, loopUbs,
        [&](const KrnlBuilder &createKrnl, ValueRange outerIndices) {
          MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
              createKrnl);
          // Create temp, single scalar, no need for default alignment.
          // Alloca is ok here as its for a scalar, and in the generic version
          // of GEMM.
          Value red = create.mem.alloca(MemRefType::get({}, elementType));
          // Set to zero.
          create.krnl.store(zeroVal, red);
          // Inner loop.
          create.krnl.iterate({}, innerLoopDef, {}, {},
              [&](const KrnlBuilder &createKrnl, ValueRange innerIndex) {
                Value i(outerIndices[0]), j(outerIndices[1]), k(innerIndex[0]);
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);
                // Handle transposed accesses.
                SmallVector<Value, 2> aAccess, bAccess;
                if (adaptor.getTransA() != 0)
                  aAccess = {k, i};
                else
                  aAccess = {i, k};
                if (adaptor.getTransB() != 0)
                  bAccess = {j, k};
                else
                  bAccess = {k, j};
                // Perform the reduction by adding a*b to reduction.
                Value aVal = create.krnl.load(A, aAccess);
                Value bVal = create.krnl.load(B, bAccess);
                Value tmp = create.math.mul(aVal, bVal);
                Value rVal = create.krnl.load(red);
                create.krnl.store(create.math.add(tmp, rVal), red);
              });
          // Handle alpha/beta coefficients.
          IndexExprScope innerScope(create.krnl, shapeHelper.getScope());
          Value res = create.math.mul(alphaVal, create.krnl.load(red));
          if (shapeHelper.hasBias) {
            SmallVector<Value, 2> cAccess;
            for (int x = 2 - shapeHelper.cRank; x < 2; ++x) {
              // If dim > 1, use loop index, otherwise broadcast on 0's element.
              DimIndexExpr dim(shapeHelper.cDims[x]);
              cAccess.emplace_back(
                  IndexExpr::select(dim > 1, DimIE(outerIndices[x]), 0)
                      .getValue());
            }
            Value c = create.krnl.load(adaptor.getC(), cAccess);
            res = create.math.add(res, create.math.mul(betaVal, c));
          }
          create.krnl.store(res, R, outerIndices);
        });
  }

  //===--------------------------------------------------------------------===//
  // Small-N Gemm.
  //
  // Y[M,N] = alpha * A[M,K] x op(B) + beta * C, for a small static number of
  // output columns N and a static K; M may be dynamic. This is the shape of
  // the small linear heads of GNNs (e.g. [M,96] x [96,12] and [M,12] x [12,2]).
  // The tiled kernel below vectorizes along N in 16-wide register tiles, so
  // with N < 16 it runs scalar, and it parallelizes over N, a single tile.
  //
  // Here every output is a dot product of a row of A with a row of B taken as
  // [N,K]: B itself with transB, a transposed copy of B otherwise (N x K
  // elements, against M x N x K multiply-adds). Both rows are contiguous along
  // K, so the kernel vectorizes along K. It processes R rows at a time, keeping
  // their R x N vector accumulators in registers (each loaded vector of B then
  // serves R rows), reduces them once per row, applies alpha and the bias in
  // registers, and stores each row. Row blocks are split across threads.

  // Largest N handled by the small-N path.
  static constexpr int64_t smallNMaxN = 16;

  // Rows per block: R x N accumulators, plus R vectors of A and one of B,
  // must fit in the vector registers, with a few to spare.
  static int64_t smallNRowsPerBlock(int64_t N) {
    int64_t regs = VectorMachineSupport::getArchVectorRegisterNum();
    return std::clamp<int64_t>((regs - 4) / (N + 1), 1, 8);
  }

  bool isSmallNGemm(ONNXGemmOpAdaptor &adaptor, Type elementType,
      ONNXGemmOpShapeHelper &shapeHelper) const {
    if (disableGemmSmallN || !enableSIMD || DEBUG_SIMD_OFF)
      return false;
    // SIMD along K needs a vector length larger than one.
    if (!elementType.isF32() ||
        VectorMachineSupport::getArchVectorLength(elementType) <= 1)
      return false;
    // Rows of A must be contiguous along K.
    if (adaptor.getTransA())
      return false;
    IndexExpr N = shapeHelper.getOutputDims()[1];
    IndexExpr K = shapeHelper.aDims[1];
    if (!N.isLiteral() || !K.isLiteral())
      return false;
    // One row's N accumulators, plus a vector of A and one of B, must fit in
    // the vector registers (16 of them on some targets).
    int64_t maxN = std::min(
        smallNMaxN, VectorMachineSupport::getArchVectorRegisterNum() - 3);
    return N.getLiteral() >= 1 && N.getLiteral() <= maxN && K.getLiteral() >= 1;
  }

  void smallNGemm(Operation *op, ONNXGemmOpAdaptor &adaptor, Type elementType,
      ONNXGemmOpShapeHelper &shapeHelper, Value alloc, Value zeroVal,
      Value alphaVal, Value betaVal, ConversionPatternRewriter &rewriter,
      Location loc, bool enableParallel) const {
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder, VectorBuilder>
        create(rewriter, loc);
    Value A(adaptor.getA()), B(adaptor.getB()), C(adaptor.getC()), Y(alloc);
    IndexExpr M = shapeHelper.getOutputDims()[0];
    int64_t N = shapeHelper.getOutputDims()[1].getLiteral();
    int64_t K = shapeHelper.aDims[1].getLiteral();
    int64_t VL = create.vec.getArchVectorLength(elementType);
    int64_t kVecs = K / VL;          // Full vectors along K.
    int64_t kTailStart = kVecs * VL; // Scalar tail [kTailStart, K).
    int64_t R = smallNRowsPerBlock(N);
    float alphaLit = adaptor.getAlpha().convertToFloat();
    float betaLit = adaptor.getBeta().convertToFloat();
    bool hasBias = shapeHelper.hasBias && betaLit != 0.0;
    VectorType vecType = VectorType::get({VL}, elementType);
    onnxToKrnlSimdReport(op, /*successful*/ true, VL, K,
        "small-N gemm with N=" + std::to_string(N) + " and simd along K");

    // A bias that is the same for every row and contiguous along N can be
    // loaded as vectors; any other one (including a rank-0 scalar) is loaded
    // element by element.
    bool biasIsRowVector = false;
    if (hasBias && shapeHelper.cRank >= 1) {
      const DimsExpr &cDims = shapeHelper.cDims;
      int64_t cRank = shapeHelper.cRank;
      biasIsRowVector =
          cDims[cRank - 1].isLiteral() && cDims[cRank - 1].getLiteral() == N &&
          (cRank == 1 || (cDims[0].isLiteral() && cDims[0].getLiteral() == 1));
    }

    // B as [N,K]: as stored with transB, else a transposed copy.
    Value BT = B;
    if (!adaptor.getTransB()) {
      BT = create.mem.alignedAlloc(
          MemRefType::get({N, K}, elementType), BUFFER_ALIGN);
      ValueRange loops = create.krnl.defineLoops(2);
      create.krnl.iterateIE(loops, loops, {LitIE(0), LitIE(0)},
          {LitIE(N), LitIE(K)},
          [&](const KrnlBuilder &createKrnl, ValueRange jk) {
            Value val = createKrnl.load(B, {jk[1], jk[0]});
            createKrnl.store(val, BT, {jk[0], jk[1]});
          });
    }

    // Compute rows [row0, row0 + numRows), numRows <= R a literal.
    auto genRows = [&](const KrnlBuilder &createKrnl, Value row0,
                       int64_t numRows) {
      MultiDialectBuilder<KrnlBuilder, MathBuilder, VectorBuilder> create(
          createKrnl);
      IndexExprScope scope(create.krnl, shapeHelper.getScope());
      auto cst = [&](int64_t v) { return create.math.constantIndex(v); };
      SmallVector<Value, 8> rows;
      for (int64_t r = 0; r < numRows; ++r)
        rows.emplace_back(create.math.add(row0, cst(r)));
      Value zeroVec = create.vec.broadcast(vecType, zeroVal);

      // Accumulate the numRows x N dot products, VL elements of K at a time.
      // acc[r * N + j] holds VL partial sums of Y[row r, j].
      SmallVector<Value, 32> acc;
      if (kVecs > 0) {
        SmallVector<Value, 32> inits(numRows * N, zeroVec);
        ValueRange kLoop = create.krnl.defineLoops(1);
        KrnlIterateOp kIterate = create.krnl.iterate(kLoop, kLoop, {cst(0)},
            {cst(kVecs)}, inits,
            [&](const KrnlBuilder &createKrnl, ValueRange kInd,
                ValueRange iterArgs) {
              MultiDialectBuilder<KrnlBuilder, MathBuilder, VectorBuilder>
                  create(createKrnl);
              Value k = create.math.mul(kInd[0], cst(VL));
              SmallVector<Value, 8> aVecs;
              for (int64_t r = 0; r < numRows; ++r)
                aVecs.emplace_back(create.vec.load(vecType, A, {rows[r], k}));
              SmallVector<Value, 32> next(iterArgs.begin(), iterArgs.end());
              for (int64_t j = 0; j < N; ++j) {
                Value bVec = create.vec.load(vecType, BT, {cst(j), k});
                for (int64_t r = 0; r < numRows; ++r)
                  next[r * N + j] =
                      create.vec.fma(aVecs[r], bVec, next[r * N + j]);
              }
              create.krnl.yield(next);
            });
        acc.append(kIterate.getResults().begin(), kIterate.getResults().end());
      }

      // Bias of Y[row, j], broadcasting C as ONNX allows.
      auto loadBias = [&](Value row, int64_t j) -> Value {
        SmallVector<Value, 2> cAccess;
        for (int64_t x = 2 - shapeHelper.cRank; x < 2; ++x) {
          DimIndexExpr dim(shapeHelper.cDims[x]);
          Value ind = (x == 0) ? row : cst(j);
          cAccess.emplace_back(
              IndexExpr::select(dim > 1, DimIE(ind), 0).getValue());
        }
        return create.krnl.load(C, cAccess);
      };

      for (int64_t r = 0; r < numRows; ++r) {
        Value row = rows[r];
        int64_t j = 0;
        // Groups of VL outputs: VL horizontal sums at once, one vector store.
        for (; kVecs > 0 && j + VL <= N; j += VL) {
          SmallVector<Value, 8> sums;
          create.vec.multiReduction(
              ArrayRef<Value>(acc).slice(r * N + j, VL),
              [&](Value a, Value b) { return create.math.add(a, b); }, sums);
          Value sum = sums[0];
          for (int64_t t = kTailStart; t < K; ++t) {
            Value aT = create.vec.broadcast(
                vecType, create.krnl.load(A, {row, cst(t)}));
            Value bT = zeroVec;
            for (int64_t l = 0; l < VL; ++l)
              bT = create.vec.insertElement(
                  bT, create.krnl.load(BT, {cst(j + l), cst(t)}), l);
            sum = create.vec.fma(aT, bT, sum);
          }
          if (alphaLit != 1.0)
            sum = create.math.mul(sum, create.vec.broadcast(vecType, alphaVal));
          if (hasBias) {
            Value cVec;
            if (biasIsRowVector) {
              SmallVector<Value, 2> cAccess(shapeHelper.cRank - 1, cst(0));
              cAccess.emplace_back(cst(j));
              cVec = create.vec.load(vecType, C, cAccess);
            } else {
              cVec = zeroVec;
              for (int64_t l = 0; l < VL; ++l)
                cVec = create.vec.insertElement(cVec, loadBias(row, j + l), l);
            }
            if (betaLit != 1.0)
              cVec =
                  create.math.mul(cVec, create.vec.broadcast(vecType, betaVal));
            sum = create.math.add(sum, cVec);
          }
          create.vec.store(sum, Y, {row, cst(j)});
        }
        // Remaining outputs one at a time.
        for (; j < N; ++j) {
          Value sum = (kVecs > 0) ? create.vec.reduction(
                                        VectorBuilder::CombiningKind::ADD,
                                        acc[r * N + j])
                                  : zeroVal;
          for (int64_t t = kTailStart; t < K; ++t) {
            Value aT = create.krnl.load(A, {row, cst(t)});
            Value bT = create.krnl.load(BT, {cst(j), cst(t)});
            sum = create.math.add(sum, create.math.mul(aT, bT));
          }
          if (alphaLit != 1.0)
            sum = create.math.mul(sum, alphaVal);
          if (hasBias) {
            Value c = loadBias(row, j);
            if (betaLit != 1.0)
              c = create.math.mul(c, betaVal);
            sum = create.math.add(sum, c);
          }
          create.krnl.store(sum, Y, {row, cst(j)});
        }
      }
    };

    // Blocks of R rows, split across threads: whole rows per thread, so each
    // writes a contiguous range of Y. One iteration does R x N dot products
    // of K elements, plus their reductions and epilogue.
    IndexExpr numBlocks = M.floorDiv(R);
    ValueRange blockLoop = create.krnl.defineLoops(1);
    auto plan = KrnlParallelPlan::noCollapse(blockLoop, /*first*/ 0,
        /*last excl*/ 1, {.minTripCount = 4, .bodyCost = R * N * (K + 4)});
    if (enableParallel)
      plan.tryCreateParallel(
          create.krnl, op, "GEMM small N row blocks", {LitIE(0)}, {numBlocks});
    create.krnl.iterateIE(blockLoop, plan.optimizedLoopDef(), {LitIE(0)},
        {numBlocks}, [&](const KrnlBuilder &createKrnl, ValueRange ind) {
          MathBuilder createMath(createKrnl);
          genRows(createKrnl,
              createMath.mul(ind[0], createMath.constantIndex(R)), R);
        });
    // The last M % R rows, one at a time.
    if (R > 1 && !(M.isLiteral() && M.getLiteral() % R == 0)) {
      ValueRange tailLoop = create.krnl.defineLoops(1);
      create.krnl.iterateIE(tailLoop, tailLoop, {numBlocks * R}, {M},
          [&](const KrnlBuilder &createKrnl, ValueRange ind) {
            genRows(createKrnl, ind[0], 1);
          });
    }
  }

  void tiledTransposedGemm(Operation *op, ONNXGemmOpAdaptor &adaptor,
      Type elementType, ONNXGemmOpShapeHelper &shapeHelper, Value alloc,
      Value zeroVal, Value alphaVal, Value betaVal,
      ConversionPatternRewriter &rewriter, Location loc,
      bool enableParallel) const {

    // R is result (alloc).
    Value A(adaptor.getA()), B(adaptor.getB()), R(alloc);
    bool aTrans = adaptor.getTransA();
    bool bTrans = adaptor.getTransB();
    IndexExpr I = shapeHelper.getOutputDims()[0];
    IndexExpr J = shapeHelper.getOutputDims()[1];
    IndexExpr K = shapeHelper.aDims[1]; // aDims are already transposed.
    LiteralIndexExpr zeroIE(0);
    Value z = zeroIE.getValue();

    // Initialize alloc/R to zero.
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder> create(rewriter, loc);
    create.krnl.memset(R, zeroVal);

    // Prepare for the computations.
    // 1) Define blocking, with simdization along the j axis.
    const int64_t iCacheTile(32), jCacheTile(64), kCacheTile(256);
    const int64_t iRegTile(4), jRegTile(16);

    bool unrollAndJam = DEBUG_UNROLL_OFF ? false : true;
    // Simdize with jRegTile as the vector length.
    bool simdize = DEBUG_SIMD_OFF ? false : enableSIMD;
    if (!simdize)
      onnxToKrnlSimdReport(op, /*successful*/ false, /*vl*/ 0,
          /*trip count*/ 0, "simd disabled");

    bool mustTileR = false;
    if (!J.isLiteral()) {
      // Assume large J, will simdize, but since simdized dimension must be a
      // multiple of the vector length, we must tile C into a smaller block of
      // known dimensions that are compatible with SIMD.
      mustTileR = true;
      if (simdize)
        onnxToKrnlSimdReport(op, /*successful*/ true, /*vl*/ jRegTile,
            /*trip count*/ -1, "simd for copied tiles due to runtime j dim");
    } else {
      int64_t jVal = J.getLiteral();
      if (jVal < jRegTile) {
        // Very small computation, give up on SIMD.
        if (simdize)
          onnxToKrnlSimdReport(op, /*successful*/ false, /*vl*/ 0,
              /*trip count*/ jVal, "no simd because of small j trip count");
        simdize = false;
      } else if (jVal % jRegTile != 0) {
        // Unfortunately, J is not divisible by the vector length. Could try
        // to change the vector length, but right now, just go to buffering.
        mustTileR = true;
        if (simdize)
          onnxToKrnlSimdReport(op, /*successful*/ true, /*vl*/ jRegTile,
              /*trip count*/ jVal, "simd for copied tiles due to j dim");
      } else {
        // Best of all world, large computation, of sizes compatible with vector
        // length.
        if (simdize)
          onnxToKrnlSimdReport(op, /*successful*/ true, /*vl*/ jRegTile,
              /*trip count*/ jVal, "simd directly in output tiles");
      }
    }

    // 2) Alloc data for tiles.
    MemRefType aTileType =
        MemRefType::get({iCacheTile, kCacheTile}, elementType);
    MemRefType bTileType =
        MemRefType::get({kCacheTile, jCacheTile}, elementType);
    SmallVector<IndexExpr, 1> empty;

    // 3) introduce the loops and permute them
    // I, J, K loop.
    ValueRange origLoop = create.krnl.defineLoops(3);
    Value ii(origLoop[0]), jj(origLoop[1]), kk(origLoop[2]);
    // Tile I.
    ValueRange iCacheBlock = create.krnl.block(ii, iCacheTile);
    ValueRange iRegBlock = create.krnl.block(iCacheBlock[1], iRegTile);
    Value ii1(iCacheBlock[0]), ii2(iRegBlock[0]), ii3(iRegBlock[1]);
    // Tile J.
    ValueRange jCacheBlock = create.krnl.block(jj, jCacheTile);
    ValueRange jRegBlock = create.krnl.block(jCacheBlock[1], jRegTile);
    Value jj1(jCacheBlock[0]), jj2(jRegBlock[0]), jj3(jRegBlock[1]);
    // Tile K.
    ValueRange kCacheBlock = create.krnl.block(kk, kCacheTile);
    Value kk1(kCacheBlock[0]), kk2(kCacheBlock[1]);

    // If we must tile the result R, then we put I & J in the outermost.
    // Otherwise, we follow the more traditional scheme of having J & K in the
    // outermost.
    if (mustTileR) {
      // (cache) ii1 jj1 kk1,    (reg) jj2, ii2,    (matmul) ii3, jj3, kk3
      create.krnl.permute({ii1, ii2, ii3, jj1, jj2, jj3, kk1, kk2},
          {/*i*/ 0, 4, 5, /*j*/ 1, 3, 6, /*k*/ 2, 7});
      // Blocked/permuted refs can never collapse. Plan holds the iterate's own
      // list; the window restricts the choice to the outermost ref.
      auto plan = KrnlParallelPlan::noCollapse({ii1, jj1}, /*first*/ 0,
          /*last excl*/ 1, {.minTripCount = 4 * iCacheTile});
      if (enableParallel)
        plan.tryCreateParallel(
            create.krnl, op, "GEMM tiled copy I parallel", {zeroIE}, {I});
      // Compute: A[i, k] * b[k, j] -> R[i, j])
      create.krnl.iterateIE({ii, jj, kk}, plan.optimizedLoopDef(),
          {zeroIE, zeroIE, zeroIE}, {I, J, K},
          [&](const KrnlBuilder &createKrnl, ValueRange i1_j1_indices) {
            Value i1(i1_j1_indices[0]), j1(i1_j1_indices[1]);
            // If parallel, will stay inside, otherwise will migrate out.
            // Since they are not in an if structure, migration out is not an
            // issue.
            Value aBuff = create.mem.alignedAlloc(aTileType, BUFFER_ALIGN);
            Value bBuff = create.mem.alignedAlloc(bTileType, BUFFER_ALIGN);
            Value rBuff = create.mem.alignedAlloc(aTileType, BUFFER_ALIGN);
            createKrnl.copyToBuffer(rBuff, R, {i1, j1}, zeroVal, false);
            createKrnl.iterateIE({}, {kk1}, {}, {},
                [&](const KrnlBuilder &createKrnl, ValueRange k1_index) {
                  Value k1(k1_index[0]);
                  if (aTrans)
                    createKrnl.copyToBuffer(aBuff, A, {k1, i1}, zeroVal, true);
                  else
                    createKrnl.copyToBuffer(aBuff, A, {i1, k1}, zeroVal, false);
                  if (bTrans)
                    createKrnl.copyToBuffer(bBuff, B, {j1, k1}, zeroVal, true);
                  else
                    createKrnl.copyToBuffer(bBuff, B, {k1, j1}, zeroVal, false);
                  createKrnl.iterate({}, {jj2, ii2}, {}, {},
                      [&](const KrnlBuilder &createKrnl,
                          ValueRange j2_i2_indices) {
                        Value j2(j2_i2_indices[0]), i2(j2_i2_indices[1]);
                        ArrayRef<int64_t> empty;
                        createKrnl.matmul(aBuff, {i1, k1}, bBuff, {k1, j1},
                            rBuff, {i1, j1},
                            /*loops*/ {ii3, jj3, kk2},
                            /*compute start*/ {i2, j2, k1},
                            /*ubs*/ {I.getValue(), J.getValue(), K.getValue()},
                            /*compute tile*/ {iRegTile, jRegTile, kCacheTile},
                            /* a/b/c tiles*/ empty, empty, empty, simdize,
                            unrollAndJam, false);
                      });
                });
            createKrnl.copyFromBuffer(rBuff, R, {i1, j1});
          });

    } else {
      // Does not have to tile the result.
      // (cache) jj1 kk1, ii1, (reg) jj2, ii2, (matmul) ii3, jj3, kk3
      // Krnl Rule: put all the values in the permute, including the ones that
      // are not iterated over explicitly. All of the same derived (tiled)
      // variable must be consecutive, and different original variables must be
      // ordered in the same permute order. Js must be first as the outermost
      // level is a j, then all the Ks, then all the Is.
      create.krnl.permute({jj1, jj2, jj3, kk1, kk2, ii1, ii2, ii3},
          {/*j*/ 0, 3, 5, /*k*/ 1, 6, /*i*/ 2, 4, 7});
      // As above: blocked/permuted refs, plan over the iterate's list.
      auto plan = KrnlParallelPlan::noCollapse({jj1, kk1}, /*first*/ 0,
          /*last excl*/ 1, {.minTripCount = 4 * jCacheTile});
      if (enableParallel)
        plan.tryCreateParallel(
            create.krnl, op, "GEMM tiled no copy J parallel", {zeroIE}, {J});
      // Compute: A[i, k] * b[k, j] -> R[i, j])
      // Krnl Rule: must put all the iter bounds at once, but can only put the
      // "not currently used ones" like ii here last. Gave an error when ii was
      // listed first.
      create.krnl.iterateIE({jj, kk, ii}, plan.optimizedLoopDef(),
          {zeroIE, zeroIE, zeroIE}, {J, K, I},
          [&](const KrnlBuilder &createKrnl, ValueRange j1_k1_indices) {
            Value j1(j1_k1_indices[0]), k1(j1_k1_indices[1]);
            // If parallel, it will stay inside, otherwise it will migrate out.
            // Since allocs are not in an if structure, migration is not an
            // issue.
            Value aBuff = create.mem.alignedAlloc(aTileType, BUFFER_ALIGN);
            Value bBuff = create.mem.alignedAlloc(bTileType, BUFFER_ALIGN);
            if (bTrans)
              createKrnl.copyToBuffer(bBuff, B, {j1, k1}, zeroVal, true);
            else
              createKrnl.copyToBuffer(bBuff, B, {k1, j1}, zeroVal, false);
            createKrnl.iterateIE({}, {ii1}, {}, {},
                [&](const KrnlBuilder &createKrnl, ValueRange i1_index) {
                  Value i1(i1_index[0]);
                  if (aTrans)
                    createKrnl.copyToBuffer(aBuff, A, {k1, i1}, zeroVal, true);
                  else
                    createKrnl.copyToBuffer(aBuff, A, {i1, k1}, zeroVal, false);
                  createKrnl.iterate({}, {jj2, ii2}, {}, {},
                      [&](const KrnlBuilder &createKrnl,
                          ValueRange j2_i2_indices) {
                        Value j2(j2_i2_indices[0]), i2(j2_i2_indices[1]);
                        createKrnl.matmul(aBuff, {i1, k1}, bBuff, {k1, j1}, R,
                            {z, z},
                            /*loops*/ {ii3, jj3, kk2},
                            /*compute start*/ {i2, j2, k1},
                            /*ubs*/ {I.getValue(), J.getValue(), K.getValue()},
                            /*compute tile*/ {iRegTile, jRegTile, kCacheTile},
                            /* a/b/c tiles*/ {}, {}, {}, simdize, unrollAndJam,
                            false);
                      });
                });
          });
    }

    // Perform the alpha/beta computations.
    float alphaLit = adaptor.getAlpha().convertToFloat();
    float betaLit = adaptor.getBeta().convertToFloat();
    if (alphaLit == 1.0 && (betaLit == 0.0 || !shapeHelper.hasBias)) {
      // No need for the multiply/add.
      return;
    }
    ValueRange outerLoops = create.krnl.defineLoops(2);
    auto plan = KrnlParallelPlan::noCollapse(outerLoops, /*first*/ 0,
        /*last excl*/ 1, {.minTripCount = 16});
    if (enableParallel)
      plan.tryCreateParallel(create.krnl, op,
          "GEMM alpha/beta epilogue outer loop", {zeroIE}, {I});
    create.krnl.iterateIE(outerLoops, plan.optimizedLoopDef(), {zeroIE, zeroIE},
        {I, J}, [&](const KrnlBuilder &createKrnl, ValueRange outerIndices) {
          // Handle alpha/beta coefficients.
          Value res = createKrnl.load(R, outerIndices);
          MathBuilder createMath(createKrnl);
          if (alphaLit != 1.0)
            res = createMath.mul(alphaVal, res);
          if (shapeHelper.hasBias) {
            IndexExprScope innerScope(createKrnl, shapeHelper.getScope());
            SmallVector<Value, 2> cAccess;
            for (int x = 2 - shapeHelper.cRank; x < 2; ++x) {
              // If dim > 1, use loop index, otherwise broadcast on 0's element.
              DimIndexExpr dim(shapeHelper.cDims[x]);
              cAccess.emplace_back(
                  IndexExpr::select(dim > 1, DimIE(outerIndices[x]), 0)
                      .getValue());
            }
            Value c = createKrnl.load(adaptor.getC(), cAccess);
            if (betaLit != 1.0)
              c = createMath.mul(betaVal, c);
            res = createMath.add(res, c);
          }
          createKrnl.store(res, R, outerIndices);
        });
  }

  LogicalResult matchAndRewrite(GemmOp gemmOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = gemmOp.getOperation();
    ValueRange operands = adaptor.getOperands();
    Location loc = ONNXLoc<GemmOp>(op);

    // Get shape.
    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    ONNXGemmOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType =
        this->typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Insert an allocation and deallocation for the output of this operation.
    Type elementType = outputMemRefType.getElementType();
    Value alloc = create.mem.alignedAlloc(
        outputMemRefType, shapeHelper.getOutputDims(), BUFFER_ALIGN);

    // Get the constants: zero, alpha,and beta.
    float alphaLit = adaptor.getAlpha().convertToFloat();
    float betaLit = adaptor.getBeta().convertToFloat();
    Value alpha = create.math.constant(elementType, alphaLit);
    Value beta = create.math.constant(elementType, betaLit);
    Value zero = create.math.constant(elementType, 0);

    LLVM_DEBUG({
      if (DEBUG_SIMD_OFF)
        llvm::dbgs() << "Gemm simd off\n";
      if (DEBUG_UNROLL_OFF)
        llvm::dbgs() << "Gemm unroll off\n";
      if (DEBUG_OPTIMIZED_OFF)
        llvm::dbgs() << "Gemm optimized path off\n";

      bool aTrans = adaptor.getTransA();
      bool bTrans = adaptor.getTransB();
      if (IndexExpr::isLiteral(shapeHelper.aDims) &&
          IndexExpr::isLiteral(shapeHelper.bDims) &&
          IndexExpr::isLiteral(shapeHelper.cDims)) {
        int64_t cDim0 = shapeHelper.hasBias ? shapeHelper.cDims[0].getLiteral()
                                            : ShapedType::kDynamic;
        int64_t cDim1 = shapeHelper.hasBias ? shapeHelper.cDims[1].getLiteral()
                                            : ShapedType::kDynamic;
        llvm::dbgs() << "OP-STATS: gemm of size I/J/K, "
                     << shapeHelper.aDims[0].getLiteral() << ","
                     << shapeHelper.bDims[1].getLiteral() << ","
                     << shapeHelper.aDims[1].getLiteral()
                     << (aTrans ? ", a trans" : "")
                     << (bTrans ? ", b trans" : "") << ", alpha " << alphaLit
                     << (alphaLit == 1.0 ? " (skip)" : "") << ", beta "
                     << betaLit << (betaLit == 1.0 ? " (skip)" : "") << ", c, "
                     << cDim0 << ", " << cDim1 << "\n";
      } else {
        llvm::dbgs() << "OP-STATS: gemm of unknown sizes "
                     << (aTrans ? ", a trans" : "")
                     << (bTrans ? ", b trans" : "") << ", alpha " << alphaLit
                     << ", beta " << betaLit << "\n";
      }
    });

    if (enableTiling && !DEBUG_OPTIMIZED_OFF &&
        isSmallNGemm(adaptor, elementType, shapeHelper)) {
      smallNGemm(op, adaptor, elementType, shapeHelper, alloc, zero, alpha,
          beta, rewriter, loc, enableParallel);
    } else if (enableTiling && !DEBUG_OPTIMIZED_OFF) {
      tiledTransposedGemm(op, adaptor, elementType, shapeHelper, alloc, zero,
          alpha, beta, rewriter, loc, enableParallel);
    } else {
      genericGemm(op, adaptor, elementType, shapeHelper, alloc, zero, alpha,
          beta, rewriter, loc, enableParallel);
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXGemmOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableTiling,
    bool enableSIMD, bool enableParallel) {
  patterns.insert<ONNXGemmOpLowering<ONNXGemmOp>>(
      typeConverter, ctx, enableTiling, enableSIMD, enableParallel);
}

} // namespace onnx_mlir
