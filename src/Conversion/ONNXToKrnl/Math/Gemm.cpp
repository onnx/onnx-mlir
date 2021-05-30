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

#include "mlir/Dialect/MemRef/EDSC/Intrinsics.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

// Used to trace which op are used, good for profiling apps.
#define TRACE 0
#define DEBUG_SIMD_OFF 0
#define DEBUG_UNROLL_OFF 0
#define DEBUG_OPTIMIZED_OFF 0
#define DEBUG_GLOBAL_ALLOC_FREE 1

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
    // Scope for krnl EDSC ops
    using namespace mlir::edsc;
    using namespace mlir::edsc::intrinsics;
    // ScopedContext scope(rewriter, loc);

    // R is result (alloc).
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), R(alloc);
    MemRefBoundsCapture rBound(R);

    // Outer loops.
    ValueRange outerLoops = krnl_define_loop(2);
    krnl_iterate(
        outerLoops, rBound.getLbs(), rBound.getUbs(), {}, [&](ValueRange args) {
          // Outer loop indices.
          ValueRange outerIndices = krnl_get_induction_var_value(outerLoops);
          // Create temp and set to zero.
          Value red = memref_alloca(MemRefType::get({}, elementType));
          SmallVector<Value, 2> redAccess; // Empty.
          krnl_store(zeroVal, red, redAccess);
          // Inner loop
          ValueRange innerLoop = krnl_define_loop(1);
          Value lb = std_constant_index(0);
          Value ub = shapeHelper.aDims[1].getValue();
          krnl_iterate(innerLoop, {lb}, {ub}, {}, [&](ValueRange args) {
            ValueRange innerIndex = krnl_get_induction_var_value(innerLoop);
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
            Value tmp = std_mulf(krnl_load(A, aAccess), krnl_load(B, bAccess));
            krnl_store(
                std_addf(tmp, krnl_load(red, redAccess)), red, redAccess);
          });
          // Handle alpha/beta coefficients.
          Value res = std_mulf(alphaVal, krnl_load(red, redAccess));
          if (shapeHelper.hasBias) {
            SmallVector<IndexExpr, 2> cAccess;
            for (int x = 2 - shapeHelper.cRank; x < 2; ++x) {
              // If dim > 1, use loop index, otherwise broadcast on 0's element.
              DimIndexExpr dim(shapeHelper.cDims[x]);
              cAccess.emplace_back(
                  IndexExpr::select(dim > 1, DimIndexExpr(outerIndices[x]), 0));
            }
            Value c = krnl_load(operandAdaptor.C(), cAccess);
            res = std_addf(res, std_mulf(betaVal, c));
          }
          krnl_store(res, R, outerIndices);
        });
  }

  void tiledTransposedGemm(ONNXGemmOp &gemmOp,
      ONNXGemmOpAdaptor &operandAdaptor, Type elementType,
      ONNXGemmOpShapeHelper &shapeHelper, Value alloc, Value zeroVal,
      Value alphaVal, Value betaVal, ConversionPatternRewriter &rewriter,
      Location loc) const {
    // Scope for krnl EDSC ops
    using namespace mlir::edsc;
    using namespace mlir::edsc::intrinsics;

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
    ValueRange zeroLoop = krnl_define_loop(2);
    krnl_iterate_ie(zeroLoop, {zero, zero}, {I, J}, {}, [&](ValueRange args) {
      ValueRange indices = krnl_get_induction_var_value(zeroLoop);
      krnl_store(zeroVal, R, indices);
    });

    // Prepare for the computations.
    // 1) Define blocking, with simdization along the j axis.
    const int64_t iCacheTile(64), jCacheTile(128), kCacheTile(512);
    const int64_t iRegTile(4), jRegTile(8);

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
    // IntegerAttr alignAttr =
    //    IntegerAttr::get(IntegerType::get(rewriter, 64), 128);
    MemRefType aTileType =
        MemRefType::get({iCacheTile, kCacheTile}, elementType);
    MemRefType bTileType =
        MemRefType::get({kCacheTile, jCacheTile}, elementType);
    MemRefType rTileType =
        MemRefType::get({iCacheTile, jCacheTile}, elementType);
#if DEBUG_GLOBAL_ALLOC_FREE
    SmallVector<IndexExpr, 1> empty;
    Value aBuff = insertAllocAndDeallocSimple(
        rewriter, gemmOp, aTileType, loc, empty, true, BUFFER_ALIGN);
    Value bBuff = insertAllocAndDeallocSimple(
        rewriter, gemmOp, bTileType, loc, empty, true, BUFFER_ALIGN);
    Value rBuff;
    if (mustTileR)
      rBuff = insertAllocAndDeallocSimple(
          rewriter, gemmOp, aTileType, loc, empty, true, BUFFER_ALIGN);
#else
    ValueRange empty;
    IntegerAttr alignAttr = rewriter.getI64IntegerAttr(BUFFER_ALIGN);
    Value aBuff = memref_alloc(aTileType, empty, alignAttr);
    Value bBuff = memref_alloc(bTileType, empty, alignAttr);
    Value rBuff;
    if (mustTileR)
      rBuff = memref_alloc(rTileType, empty, alignAttr);
#endif

    // 3) introduce the loops and permute them
    // I, J, K loop.
    ValueRange origLoop = krnl_define_loop(3);
    Value ii(origLoop[0]), jj(origLoop[1]), kk(origLoop[2]);
    // Tile I.
    ValueRange iCacheBlock = krnl_block(ii, iCacheTile);
    ValueRange iRegBlock = krnl_block(iCacheBlock[1], iRegTile);
    Value ii1(iCacheBlock[0]), ii2(iRegBlock[0]), ii3(iRegBlock[1]);
    // Tile J.
    ValueRange jCacheBlock = krnl_block(jj, jCacheTile);
    ValueRange jRegBlock = krnl_block(jCacheBlock[1], jRegTile);
    Value jj1(jCacheBlock[0]), jj2(jRegBlock[0]), jj3(jRegBlock[1]);
    // Tile K.
    ValueRange kCacheBlock = krnl_block(kk, kCacheTile);
    Value kk1(kCacheBlock[0]), kk2(kCacheBlock[1]);

    // If we must tile the result R, then we put I & J in the outermost.
    // Otherwise, we follow the more traditional scheme of having J & K in the
    // outermost.
    if (mustTileR) {
      // (cache) ii1 jj1 kk1,    (reg) jj2, ii2,    (matmul) ii3, jj3, kk3
      krnl_permute({ii1, ii2, ii3, jj1, jj2, jj3, kk1, kk2},
          {/*i*/ 0, 4, 5, /*j*/ 1, 3, 6, /*k*/ 2, 7});
      // Compute: A[i, k] * b[k, j] -> R[i, j])
      krnl_iterate_ie(
          {ii, jj}, {ii1, jj1}, {zero, zero}, {I, J}, {}, [&](ValueRange args) {
            ValueRange i1_j1_indices = krnl_get_induction_var_value({ii1, jj1});
            Value i1(i1_j1_indices[0]), j1(i1_j1_indices[1]);
            krnl_copy_to_buffer(rBuff, R, {i1, j1}, zeroVal, false);
            krnl_iterate_ie({kk}, {kk1}, {zero}, {K}, {}, [&](ValueRange args) {
              ValueRange k1_index = krnl_get_induction_var_value({kk1});
              Value k1(k1_index[0]);
              if (aTrans)
                krnl_copy_to_buffer(aBuff, A, {k1, i1}, zeroVal, true);
              else
                krnl_copy_to_buffer(aBuff, A, {i1, k1}, zeroVal, false);
              if (bTrans)
                krnl_copy_to_buffer(bBuff, B, {j1, k1}, zeroVal, true);
              else
                krnl_copy_to_buffer(bBuff, B, {k1, j1}, zeroVal, false);
              krnl_iterate({}, {jj2, ii2}, {}, {}, {}, [&](ValueRange args) {
                ValueRange j2_i2_indices =
                    krnl_get_induction_var_value({jj2, ii2});
                Value j2(j2_i2_indices[0]), i2(j2_i2_indices[1]);
                krnl_matmul(aBuff, {i1, k1}, bBuff, {k1, j1}, rBuff, {i1, j1},
                    /*loops*/ {ii3, jj3, kk2},
                    /*compute start*/ {i2, j2, k1},
                    /*ubs*/ {I.getValue(), J.getValue(), K.getValue()},
                    /*compute tile*/ {iRegTile, jRegTile, kCacheTile},
                    /* a/b/c tiles*/ {}, {}, {}, simdize, unrollAndJam, false);
              });
            });
            krnl_copy_from_buffer(rBuff, R, {i1, j1});
          });

    } else {
      // Does not have to tile the result.
      // (cache) jj1 kk1, ii1, (reg) jj2, ii2, (matmul) ii3, jj3, kk3
      krnl_permute({jj1, jj2, jj3, kk1, kk2, ii1, ii2, ii3},
          {/*j*/ 0, 3, 5, /*k*/ 1, 6, /*i*/ 2, 4, 7});
      // Compute: A[i, k] * b[k, j] -> R[i, j])
      krnl_iterate_ie(
          {jj, kk}, {jj1, kk1}, {zero, zero}, {J, K}, {}, [&](ValueRange args) {
            ValueRange j1_k1_indices = krnl_get_induction_var_value({jj1, kk1});
            Value j1(j1_k1_indices[0]), k1(j1_k1_indices[1]);
            if (bTrans)
              krnl_copy_to_buffer(bBuff, B, {j1, k1}, zeroVal, true);
            else
              krnl_copy_to_buffer(bBuff, B, {k1, j1}, zeroVal, false);
            krnl_iterate_ie({ii}, {ii1}, {zero}, {I}, {}, [&](ValueRange args) {
              ValueRange i1_index = krnl_get_induction_var_value({ii1});
              Value i1(i1_index[0]);
              if (aTrans)
                krnl_copy_to_buffer(aBuff, A, {k1, i1}, zeroVal, true);
              else
                krnl_copy_to_buffer(aBuff, A, {i1, k1}, zeroVal, false);
              krnl_iterate({}, {jj2, ii2}, {}, {}, {}, [&](ValueRange args) {
                ValueRange j2_i2_indices =
                    krnl_get_induction_var_value({jj2, ii2});
                Value j2(j2_i2_indices[0]), i2(j2_i2_indices[1]);
                krnl_matmul(aBuff, {i1, k1}, bBuff, {k1, j1}, R, {z, z},
                    /*loops*/ {ii3, jj3, kk2},
                    /*compute start*/ {i2, j2, k1},
                    /*ubs*/ {I.getValue(), J.getValue(), K.getValue()},
                    /*compute tile*/ {iRegTile, jRegTile, kCacheTile},
                    /* a/b/c tiles*/ {}, {}, {}, simdize, unrollAndJam, false);
              });
            });
          });
    }

#if DEBUG_GLOBAL_ALLOC_FREE
#else
    rewriter.create<memref::DeallocOp>(loc, aBuff);
    rewriter.create<memref::DeallocOp>(loc, bBuff);
    if (mustTileR)
      rewriter.create<memref::DeallocOp>(loc, rBuff);
#endif

    // Perform the alpha/beta computations.
    float alphaLit = gemmOp.alpha().convertToFloat();
    float betaLit = gemmOp.beta().convertToFloat();
    if (alphaLit == 1.0 && (betaLit == 0.0 || !shapeHelper.hasBias)) {
      // No need for the multiply/add.
      return;
    }
    ValueRange outerLoops = krnl_define_loop(2);
    krnl_iterate_ie(outerLoops, {zero, zero}, {I, J}, {}, [&](ValueRange args) {
      // Outer loop indices.
      ValueRange outerIndices = krnl_get_induction_var_value(outerLoops);

      // Handle alpha/beta coefficients.
      Value res = krnl_load(R, outerIndices);
      if (alphaLit != 1.0)
        res = std_mulf(alphaVal, res);
      if (shapeHelper.hasBias) {
        IndexExprScope innerScope;
        SmallVector<IndexExpr, 2> cAccess;
        for (int x = 2 - shapeHelper.cRank; x < 2; ++x) {
          // If dim > 1, use loop index, otherwise broadcast on 0's element.
          DimIndexExpr dim(shapeHelper.cDims[x]);
          cAccess.emplace_back(
              IndexExpr::select(dim > 1, DimIndexExpr(outerIndices[x]), 0));
        }
        Value c = krnl_load(operandAdaptor.C(), cAccess);
        if (betaLit != 1.0)
          c = std_mulf(betaVal, c);
        res = std_addf(res, c);
      }
      krnl_store(res, R, outerIndices);
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
    // Scope for krnl EDSC ops
    using namespace mlir::edsc;
    ScopedContext scope(rewriter, loc);

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
