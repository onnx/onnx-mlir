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

#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

// Used to trace which op are used, good for profiling apps.
#define DEBUG 0

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
          Value red = std_alloca(MemRefType::get({}, elementType));
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
              SymbolIndexExpr dim(shapeHelper.cDims[x]);
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
    // 2) Alloc data for tiles.
    MemRefType aTileType =
        MemRefType::get({iCacheTile, kCacheTile}, elementType);
    MemRefType bTileType =
        MemRefType::get({kCacheTile, jCacheTile}, elementType);
    // IntegerAttr alignAttr =
    //    IntegerAttr::get(IntegerType::get(rewriter, 64), 128);

    ValueRange empty;
    Value aBuff = std_alloc(aTileType, empty);
    Value bBuff = std_alloc(bTileType, empty);

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
    // (cache) jj1 kk1, ii1,    (reg) jj2, ii2,    (matmul) ii3, jj3, kk3
    krnl_permute({jj1, jj2, jj3, kk1, kk2, ii1, ii2, ii3},
        {/*j*/ 0, 3, 5, /*k*/ 1, 6, /*i*/ 2, 4, 7});

    // Compute: A[i, k] * b[k, j] -> R[i, j])
    krnl_iterate_ie(
        {jj, kk}, {jj1, kk1}, {zero, zero}, {J, K}, {}, [&](ValueRange args) {
          ValueRange j1_k1_indices = krnl_get_induction_var_value({jj1, kk1});
          Value j1(j1_k1_indices[0]), k1(j1_k1_indices[1]);
          if (bTrans)
            krnl_copy_to_buffer(bBuff, B, {j1, k1}, zeroVal, bTrans);
          else
            krnl_copy_to_buffer(bBuff, B, {k1, j1}, zeroVal, bTrans);
          krnl_iterate_ie({ii}, {ii1}, {zero}, {I}, {}, [&](ValueRange args) {
            ValueRange i1_index = krnl_get_induction_var_value({ii1});
            Value i1(i1_index[0]);
            if (aTrans)
              krnl_copy_to_buffer(aBuff, A, {k1, i1}, zeroVal, aTrans);
            else
              krnl_copy_to_buffer(aBuff, A, {i1, k1}, zeroVal, aTrans);
            krnl_iterate({}, {jj2, ii2}, {}, {}, {}, [&](ValueRange args) {
              ValueRange j2_i2_indices =
                  krnl_get_induction_var_value({jj2, ii2});
              Value j2(j2_i2_indices[0]), i2(j2_i2_indices[1]);
              krnl_matmul(aBuff, {i1, k1}, bBuff, {k1, j1}, R,
                  {zero.getValue(), zero.getValue()},
                  /*loops*/ {ii3, jj3, kk2},
                  /*compute start*/ {i2, j2, k1},
                  /*ubs*/ {I.getValue(), J.getValue(), K.getValue()},
                  /*compute tile*/ {iRegTile, jRegTile, kCacheTile},
                  /* a/b/c tiles*/ {}, {}, {}, true, true, false);
            });
          });
        });
    rewriter.create<DeallocOp>(loc, aBuff);
    rewriter.create<DeallocOp>(loc, bBuff);

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
    ONNXGemmOpShapeHelper shapeHelper(&gemmOp, &rewriter);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    shapeHelper.scope.debugPrint("initial scope");
    assert(succeeded(shapecomputed));
    // Scope for krnl EDSC ops
    using namespace mlir::edsc;
    ScopedContext scope(rewriter, loc);

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = outputMemRefType.getElementType();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Get the constants: zero, alpha,and beta.
    float alphaLit = gemmOp.alpha().convertToFloat();
    float betaLit = gemmOp.beta().convertToFloat();
    Value alpha = emitConstantOp(rewriter, loc, elementType, alphaLit);
    Value beta = emitConstantOp(rewriter, loc, elementType, betaLit);
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);

#if DEBUG
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

    if (true) {
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

void populateLoweringONNXGemmOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpLowering<ONNXGemmOp>>(ctx);
}
