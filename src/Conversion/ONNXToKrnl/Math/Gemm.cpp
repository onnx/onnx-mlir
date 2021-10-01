/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Gemm.cpp - Lowering Gemm Op ------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gemm Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

// Used to trace which op are used, good for profiling apps.
#define TRACE 0
#define DEBUG_SIMD_OFF 0
#define DEBUG_UNROLL_OFF 0
#define DEBUG_OPTIMIZED_OFF 0

#define BUFFER_ALIGN 128
using namespace mlir;

template <typename GemmOp>
struct ONNXGemmOpLowering : public ConversionPattern {
  ONNXGemmOpLowering(MLIRContext *ctx)
      : ConversionPattern(GemmOp::getOperationName(), 1, ctx) {}

  void genericGemm(ONNXGemmOp &gemmOp, ONNXGemmOpAdaptor &operandAdaptor,
      Type elementType, ONNXGemmOpShapeHelper &shapeHelper, Value alloc,
      Value zeroVal, Value alphaVal, Value betaVal,
      ConversionPatternRewriter &rewriter, Location loc) const {
    // R is result (alloc).
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), R(alloc);

    // Outer loops.
    KrnlBuilder createKrnl(rewriter, loc);
    ValueRange outerLoops = createKrnl.defineLoops(2);
    SmallVector<IndexExpr, 0> outerLbs(2, LiteralIndexExpr(0));
    createKrnl.iterateIE(outerLoops, outerLoops, outerLbs,
        shapeHelper.dimsForOutput(0),
        [&](KrnlBuilder &createKrnl, ValueRange outerIndices) {
          // Create temp and set to zero.
          ImplicitLocOpBuilder lb(createKrnl.getLoc(), createKrnl.getBuilder());
          // Single scalar, no need for default alignment.
          MemRefBuilder createMemRef(createKrnl);
          Value red = createMemRef.alloca(MemRefType::get({}, elementType));
          createKrnl.store(zeroVal, red);
          // Inner loop
          ValueRange innerLoop = createKrnl.defineLoops(1);
          Value innerLb = lb.create<ConstantIndexOp>(0);
          Value innerUb = shapeHelper.aDims[1].getValue();
          createKrnl.iterate(innerLoop, innerLoop, {innerLb}, {innerUb},
              [&](KrnlBuilder &createKrnl, ValueRange innerIndex) {
                Value i(outerIndices[0]), j(outerIndices[1]), k(innerIndex[0]);
                // Handle transposed accesses.
                SmallVector<Value, 2> aAccess, bAccess;
                if (gemmOp.transA() != 0)
                  aAccess = {k, i};
                else
                  aAccess = {i, k};
                if (gemmOp.transB() != 0)
                  bAccess = {j, k};
                else
                  bAccess = {k, j};
                // Perform the reduction by adding a*b to reduction.
                MathBuilder createMath(createKrnl);
                Value aVal = createKrnl.load(A, aAccess);
                Value bVal = createKrnl.load(B, bAccess);
                Value tmp = createMath.mul(aVal, bVal);
                Value rVal = createKrnl.load(red);
                createKrnl.store(createMath.add(tmp, rVal), red);
              });
          // Handle alpha/beta coefficients.
          MathBuilder createMath(createKrnl);
          // new scope
          IndexExprScope innerScope(createKrnl, shapeHelper.scope);
          Value res = createMath.mul(alphaVal, createKrnl.load(red));
          if (shapeHelper.hasBias) {
            SmallVector<Value, 2> cAccess;
            for (int x = 2 - shapeHelper.cRank; x < 2; ++x) {
              // If dim > 1, use loop index, otherwise broadcast on 0's element.
              DimIndexExpr dim(shapeHelper.cDims[x]);
              cAccess.emplace_back(
                  IndexExpr::select(dim > 1, DimIndexExpr(outerIndices[x]), 0)
                      .getValue());
            }
            Value c = createKrnl.load(operandAdaptor.C(), cAccess);
            res = createMath.add(res, createMath.mul(betaVal, c));
          }
          createKrnl.store(res, R, outerIndices);
        });
  }

  void tiledTransposedGemm(ONNXGemmOp &gemmOp,
      ONNXGemmOpAdaptor &operandAdaptor, Type elementType,
      ONNXGemmOpShapeHelper &shapeHelper, Value alloc, Value zeroVal,
      Value alphaVal, Value betaVal, ConversionPatternRewriter &rewriter,
      Location loc) const {

    // R is result (alloc).
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), R(alloc);
    bool aTrans = gemmOp.transA();
    bool bTrans = gemmOp.transB();
    IndexExpr I = shapeHelper.dimsForOutput()[0];
    IndexExpr J = shapeHelper.dimsForOutput()[1];
    IndexExpr K = shapeHelper.aDims[1]; // aDims are already transposed.
    LiteralIndexExpr zero(0);
    Value z = zero.getValue();

    // Initialize alloc/R to zero.
    KrnlBuilder createKrnl(rewriter, loc);
    createKrnl.memset(R, zeroVal);

    // Prepare for the computations.
    // 1) Define blocking, with simdization along the j axis.
    const int64_t iCacheTile(32), jCacheTile(64), kCacheTile(256);
    const int64_t iRegTile(4), jRegTile(16);

    bool unrollAndJam = DEBUG_UNROLL_OFF ? false : true;
    // Simdize with jRegTile as the vector length.
    bool simdize = DEBUG_SIMD_OFF ? false : true;

    bool mustTileR = false;
    if (!J.isLiteral()) {
      // Assume large J, will simdize, but since simdized dimension must be a
      // multiple of the vector length, we must tile C into a smaller block of
      // known dimensions that are compatible with SIMD.
      mustTileR = true;
    } else {
      int64_t jVal = J.getLiteral();
      if (jVal < jRegTile) {
        // Very small computation, give up on SIMD.
        simdize = false;
      } else if (jVal % jRegTile != 0) {
        // Unfortunately, J is not divisible by the vector length. Could try
        // to change the vector length, but right now, just go to buffering.
        mustTileR = true;
      } else {
        // Best of all world, large computation, of sizes compatible with vector
        // length.
      }
    }

    // 2) Alloc data for tiles.
    MemRefType aTileType =
        MemRefType::get({iCacheTile, kCacheTile}, elementType);
    MemRefType bTileType =
        MemRefType::get({kCacheTile, jCacheTile}, elementType);
    SmallVector<IndexExpr, 1> empty;
    Value aBuff = insertAllocAndDeallocSimple(
        rewriter, gemmOp, aTileType, loc, empty, true, BUFFER_ALIGN);
    Value bBuff = insertAllocAndDeallocSimple(
        rewriter, gemmOp, bTileType, loc, empty, true, BUFFER_ALIGN);
    Value rBuff;
    if (mustTileR)
      rBuff = insertAllocAndDeallocSimple(
          rewriter, gemmOp, aTileType, loc, empty, true, BUFFER_ALIGN);

    // 3) introduce the loops and permute them
    // I, J, K loop.
    ValueRange origLoop = createKrnl.defineLoops(3);
    Value ii(origLoop[0]), jj(origLoop[1]), kk(origLoop[2]);
    // Tile I.
    ValueRange iCacheBlock = createKrnl.block(ii, iCacheTile);
    ValueRange iRegBlock = createKrnl.block(iCacheBlock[1], iRegTile);
    Value ii1(iCacheBlock[0]), ii2(iRegBlock[0]), ii3(iRegBlock[1]);
    // Tile J.
    ValueRange jCacheBlock = createKrnl.block(jj, jCacheTile);
    ValueRange jRegBlock = createKrnl.block(jCacheBlock[1], jRegTile);
    Value jj1(jCacheBlock[0]), jj2(jRegBlock[0]), jj3(jRegBlock[1]);
    // Tile K.
    ValueRange kCacheBlock = createKrnl.block(kk, kCacheTile);
    Value kk1(kCacheBlock[0]), kk2(kCacheBlock[1]);

    // If we must tile the result R, then we put I & J in the outermost.
    // Otherwise, we follow the more traditional scheme of having J & K in the
    // outermost.
    if (mustTileR) {
      // (cache) ii1 jj1 kk1,    (reg) jj2, ii2,    (matmul) ii3, jj3, kk3
      createKrnl.permute({ii1, ii2, ii3, jj1, jj2, jj3, kk1, kk2},
          {/*i*/ 0, 4, 5, /*j*/ 1, 3, 6, /*k*/ 2, 7});
      // Compute: A[i, k] * b[k, j] -> R[i, j])
      createKrnl.iterateIE({ii, jj}, {ii1, jj1}, {zero, zero}, {I, J},
          [&](KrnlBuilder &createKrnl, ValueRange i1_j1_indices) {
            Value i1(i1_j1_indices[0]), j1(i1_j1_indices[1]);
            createKrnl.copyToBuffer(rBuff, R, {i1, j1}, zeroVal, false);
            createKrnl.iterateIE({kk}, {kk1}, {zero}, {K},
                [&](KrnlBuilder &createKrnl, ValueRange k1_index) {
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
                      [&](KrnlBuilder &createKrnl, ValueRange j2_i2_indices) {
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
      createKrnl.permute({jj1, jj2, jj3, kk1, kk2, ii1, ii2, ii3},
          {/*j*/ 0, 3, 5, /*k*/ 1, 6, /*i*/ 2, 4, 7});
      // Compute: A[i, k] * b[k, j] -> R[i, j])
      createKrnl.iterateIE({jj, kk}, {jj1, kk1}, {zero, zero}, {J, K},
          [&](KrnlBuilder &createKrnl, ValueRange j1_k1_indices) {
            Value j1(j1_k1_indices[0]), k1(j1_k1_indices[1]);
            if (bTrans)
              createKrnl.copyToBuffer(bBuff, B, {j1, k1}, zeroVal, true);
            else
              createKrnl.copyToBuffer(bBuff, B, {k1, j1}, zeroVal, false);
            createKrnl.iterateIE({ii}, {ii1}, {zero}, {I},
                [&](KrnlBuilder &createKrnl, ValueRange i1_index) {
                  Value i1(i1_index[0]);
                  if (aTrans)
                    createKrnl.copyToBuffer(aBuff, A, {k1, i1}, zeroVal, true);
                  else
                    createKrnl.copyToBuffer(aBuff, A, {i1, k1}, zeroVal, false);
                  createKrnl.iterate({}, {jj2, ii2}, {}, {},
                      [&](KrnlBuilder &createKrnl, ValueRange j2_i2_indices) {
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
    float alphaLit = gemmOp.alpha().convertToFloat();
    float betaLit = gemmOp.beta().convertToFloat();
    if (alphaLit == 1.0 && (betaLit == 0.0 || !shapeHelper.hasBias)) {
      // No need for the multiply/add.
      return;
    }
    ValueRange outerLoops = createKrnl.defineLoops(2);
    createKrnl.iterateIE(outerLoops, outerLoops, {zero, zero}, {I, J},
        [&](KrnlBuilder &createKrnl, ValueRange outerIndices) {
          // Handle alpha/beta coefficients.
          Value res = createKrnl.load(R, outerIndices);
          MathBuilder createMath(createKrnl);
          if (alphaLit != 1.0)
            res = createMath.mul(alphaVal, res);
          if (shapeHelper.hasBias) {
            IndexExprScope innerScope(createKrnl, shapeHelper.scope);
            SmallVector<Value, 2> cAccess;
            for (int x = 2 - shapeHelper.cRank; x < 2; ++x) {
              // If dim > 1, use loop index, otherwise broadcast on 0's element.
              DimIndexExpr dim(shapeHelper.cDims[x]);
              cAccess.emplace_back(
                  IndexExpr::select(dim > 1, DimIndexExpr(outerIndices[x]), 0)
                      .getValue());
            }
            Value c = createKrnl.load(operandAdaptor.C(), cAccess);
            if (betaLit != 1.0)
              c = createMath.mul(betaVal, c);
            res = createMath.add(res, c);
          }
          createKrnl.store(res, R, outerIndices);
        });
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Get shape.
    ONNXGemmOpAdaptor operandAdaptor(operands);
    ONNXGemmOp gemmOp = llvm::cast<ONNXGemmOp>(op);
    Location loc = op->getLoc();
    ONNXGemmOpShapeHelper shapeHelper(&gemmOp, rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = outputMemRefType.getElementType();
    Value alloc = insertAllocAndDeallocSimple(rewriter, op, outputMemRefType,
        loc, shapeHelper.dimsForOutput(0), (int64_t)BUFFER_ALIGN);

    // Get the constants: zero, alpha,and beta.
    float alphaLit = gemmOp.alpha().convertToFloat();
    float betaLit = gemmOp.beta().convertToFloat();
    Value alpha = emitConstantOp(rewriter, loc, elementType, alphaLit);
    Value beta = emitConstantOp(rewriter, loc, elementType, betaLit);
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);

#if TRACE
    if (DEBUG_SIMD_OFF)
      printf("Gemm simd off\n");
    if (DEBUG_UNROLL_OFF)
      printf("Gemm unroll off\n");
    if (DEBUG_OPTIMIZED_OFF)
      printf("Gemm optimized path off\n");
    bool aTrans = gemmOp.transA();
    bool bTrans = gemmOp.transB();
    if (IndexExpr::isLiteral(shapeHelper.aDims) &&
        IndexExpr::isLiteral(shapeHelper.bDims) &&
        IndexExpr::isLiteral(shapeHelper.cDims)) {
      int cDim0 = shapeHelper.hasBias ? shapeHelper.cDims[0].getLiteral() : -1;
      int cDim1 = shapeHelper.hasBias ? shapeHelper.cDims[1].getLiteral() : -1;
      printf(
          "OP-STATS: gemm of size I/J/K, %d,%d,%d%s%s, alpha %f%s, beta %f%s, "
          "c, %d, %d\n",
          (int)shapeHelper.aDims[0].getLiteral(),
          (int)shapeHelper.bDims[1].getLiteral(),
          (int)shapeHelper.aDims[1].getLiteral(), (aTrans ? ", a trans" : ""),
          (bTrans ? ", b trans" : ""), (double)alphaLit,
          (alphaLit == 1.0 ? " (skip)" : ""), (double)betaLit,
          (betaLit == 1.0 ? " (skip)" : ""), cDim0, cDim1);
    } else {
      printf("OP-STATS: gemm of unkown sizes %s%s, alpha %f, beta %f\n",
          (aTrans ? ", a trans" : ""), (bTrans ? ", b trans" : ""),
          (double)alphaLit, (double)betaLit);
    }
#endif

    if (DEBUG_OPTIMIZED_OFF) {
      genericGemm(gemmOp, operandAdaptor, elementType, shapeHelper, alloc, zero,
          alpha, beta, rewriter, loc);
    } else {
      tiledTransposedGemm(gemmOp, operandAdaptor, elementType, shapeHelper,
          alloc, zero, alpha, beta, rewriter, loc);
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXGemmOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpLowering<ONNXGemmOp>>(ctx);
}
