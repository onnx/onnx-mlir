/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Gemm.cpp - Lowering Gemm Op ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

// Used to trace which op are used, good for profiling apps.
#define DEBUG_TYPE "gemm"
#define DEBUG_SIMD_OFF 0
#define DEBUG_UNROLL_OFF 0
#define DEBUG_OPTIMIZED_OFF 0

static constexpr int BUFFER_ALIGN = 128;

using namespace mlir;

namespace onnx_mlir {

template <typename GemmOp>
struct ONNXGemmOpLowering : public ConversionPattern {
  ONNXGemmOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableTiling)
      : ConversionPattern(typeConverter, GemmOp::getOperationName(), 1, ctx),
        enableTiling(enableTiling) {}

  bool enableTiling;

  void genericGemm(ONNXGemmOp &gemmOp, ONNXGemmOpAdaptor &operandAdaptor,
      Type elementType, ONNXGemmOpShapeHelper &shapeHelper, Value alloc,
      Value zeroVal, Value alphaVal, Value betaVal,
      ConversionPatternRewriter &rewriter, Location loc) const {
    // R is result (alloc).
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), R(alloc);

    // Create all the loops at once (outerloops followed by inner loop).
    KrnlBuilder createKrnl(rewriter, loc);
    ValueRange loopDef = createKrnl.defineLoops(3);
    ValueRange outerLoopDef{loopDef[0], loopDef[1]};
    ValueRange innerLoopDef{loopDef[2]};
    SmallVector<IndexExpr, 3> loopLbs(3, LiteralIndexExpr(0));
    IndexExpr outerUb0 = shapeHelper.dimsForOutput()[0];
    IndexExpr outerUb1 = shapeHelper.dimsForOutput()[1];
    IndexExpr innerUb = shapeHelper.aDims[1];
    SmallVector<IndexExpr, 3> loopUbs{outerUb0, outerUb1, innerUb};
    // Outer loops.
    createKrnl.iterateIE(loopDef, outerLoopDef, loopLbs, loopUbs,
        [&](KrnlBuilder &createKrnl, ValueRange outerIndices) {
          // Create temp and set to zero, single scalar, no need for default
          // alignment.
          MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
              createKrnl);
          Value red = create.mem.alloca(MemRefType::get({}, elementType));
          createKrnl.store(zeroVal, red);
          // Inner loop.
          create.krnl.iterate({}, innerLoopDef, {}, {},
              [&](KrnlBuilder &createKrnl, ValueRange innerIndex) {
                Value i(outerIndices[0]), j(outerIndices[1]), k(innerIndex[0]);
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);
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
                Value aVal = create.krnl.load(A, aAccess);
                Value bVal = create.krnl.load(B, bAccess);
                Value tmp = create.math.mul(aVal, bVal);
                Value rVal = create.krnl.load(red);
                create.krnl.store(create.math.add(tmp, rVal), red);
              });
          // Handle alpha/beta coefficients.
          // new scope
          IndexExprScope innerScope(create.krnl, shapeHelper.scope);
          Value res = create.math.mul(alphaVal, createKrnl.load(red));
          if (shapeHelper.hasBias) {
            SmallVector<Value, 2> cAccess;
            for (int x = 2 - shapeHelper.cRank; x < 2; ++x) {
              // If dim > 1, use loop index, otherwise broadcast on 0's element.
              DimIndexExpr dim(shapeHelper.cDims[x]);
              cAccess.emplace_back(
                  IndexExpr::select(dim > 1, DimIndexExpr(outerIndices[x]), 0)
                      .getValue());
            }
            Value c = create.krnl.load(operandAdaptor.C(), cAccess);
            res = create.math.add(res, create.math.mul(betaVal, c));
          }
          create.krnl.store(res, R, outerIndices);
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
    LiteralIndexExpr zeroIE(0);
    Value z = zeroIE.getValue();

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
      createKrnl.iterateIE({ii, jj, kk}, {ii1, jj1}, {zeroIE, zeroIE, zeroIE},
          {I, J, K}, [&](KrnlBuilder &createKrnl, ValueRange i1_j1_indices) {
            Value i1(i1_j1_indices[0]), j1(i1_j1_indices[1]);
            createKrnl.copyToBuffer(rBuff, R, {i1, j1}, zeroVal, false);
            createKrnl.iterateIE({}, {kk1}, {}, {},
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
      // Krnl Rule: put all the values in the permute, including the ones that
      // are not iterated over explicitly. All of the same derived (tiled)
      // variable must be consecutive, and different original variables must be
      // ordered in the same permute order. Js must be first as the outermost
      // level is a j, then all the Ks, then all the Is.
      createKrnl.permute({jj1, jj2, jj3, kk1, kk2, ii1, ii2, ii3},
          {/*j*/ 0, 3, 5, /*k*/ 1, 6, /*i*/ 2, 4, 7});
      // Compute: A[i, k] * b[k, j] -> R[i, j])
      // Krnl Rule: must put all the iter bounds at once, but can only put the
      // "not currently used ones" like ii here last. Gave an error when ii was
      // listed first.
      createKrnl.iterateIE({jj, kk, ii}, {jj1, kk1}, {zeroIE, zeroIE, zeroIE},
          {J, K, I}, [&](KrnlBuilder &createKrnl, ValueRange j1_k1_indices) {
            Value j1(j1_k1_indices[0]), k1(j1_k1_indices[1]);
            if (bTrans)
              createKrnl.copyToBuffer(bBuff, B, {j1, k1}, zeroVal, true);
            else
              createKrnl.copyToBuffer(bBuff, B, {k1, j1}, zeroVal, false);
            createKrnl.iterateIE({}, {ii1}, {}, {},
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
    createKrnl.iterateIE(outerLoops, outerLoops, {zeroIE, zeroIE}, {I, J},
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
    ONNXGemmOpShapeHelper shapeHelper(&gemmOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = outputMemRefType.getElementType();
    Value alloc = insertAllocAndDeallocSimple(rewriter, op, outputMemRefType,
        loc, shapeHelper.dimsForOutput(), (int64_t)BUFFER_ALIGN);

    // Get the constants: zero, alpha,and beta.
    float alphaLit = gemmOp.alpha().convertToFloat();
    float betaLit = gemmOp.beta().convertToFloat();
    Value alpha = emitConstantOp(rewriter, loc, elementType, alphaLit);
    Value beta = emitConstantOp(rewriter, loc, elementType, betaLit);
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);

    LLVM_DEBUG({
      if (DEBUG_SIMD_OFF)
        llvm::dbgs() << "Gemm simd off\n";
      if (DEBUG_UNROLL_OFF)
        llvm::dbgs() << "Gemm unroll off\n";
      if (DEBUG_OPTIMIZED_OFF)
        llvm::dbgs() << "Gemm optimized path off\n";

      bool aTrans = gemmOp.transA();
      bool bTrans = gemmOp.transB();
      if (IndexExpr::isLiteral(shapeHelper.aDims) &&
          IndexExpr::isLiteral(shapeHelper.bDims) &&
          IndexExpr::isLiteral(shapeHelper.cDims)) {
        int cDim0 =
            shapeHelper.hasBias ? shapeHelper.cDims[0].getLiteral() : -1;
        int cDim1 =
            shapeHelper.hasBias ? shapeHelper.cDims[1].getLiteral() : -1;
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
        llvm::dbgs() << "OP-STATS: gemm of unkown sizes "
                     << (aTrans ? ", a trans" : "")
                     << (bTrans ? ", b trans" : "") << ", alpha " << alphaLit
                     << ", beta " << betaLit << "\n";
      }
    });

    if (enableTiling && !DEBUG_OPTIMIZED_OFF) {
      tiledTransposedGemm(gemmOp, operandAdaptor, elementType, shapeHelper,
          alloc, zero, alpha, beta, rewriter, loc);
    } else {
      genericGemm(gemmOp, operandAdaptor, elementType, shapeHelper, alloc, zero,
          alpha, beta, rewriter, loc);
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXGemmOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableTiling) {
  patterns.insert<ONNXGemmOpLowering<ONNXGemmOp>>(
      typeConverter, ctx, enableTiling);
}

} // namespace onnx_mlir
