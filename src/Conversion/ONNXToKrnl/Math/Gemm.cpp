/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Gemm.cpp - Lowering Gemm Op
//-------------------------===//
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

using namespace mlir;

template <typename GemmOp>
struct ONNXGemmOpLowering : public ConversionPattern {
  ONNXGemmOpLowering(MLIRContext *ctx)
      : ConversionPattern(GemmOp::getOperationName(), 1, ctx) {}

  void genericGemmV2(ONNXGemmOp &gemmOp, ONNXGemmOpAdaptor &operandAdaptor,
      Type elementType, ONNXGemmOpShapeHelper &shapeHelper, Value alloc,
      Value zeroVal, Value alphaVal, Value betaVal,
      ConversionPatternRewriter &rewriter, Location loc) const {
    // Scope for krnl EDSC ops
    using namespace mlir::edsc;
    using namespace mlir::edsc::intrinsics;
    ScopedContext scope(rewriter, loc);

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
    ScopedContext scope(rewriter, loc);

    // R is result (alloc).
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), R(alloc);
    MemRefBoundsCapture aBound(A), rBound(R);
    Value I(rBound.ub(0)), J(rBound.ub(1)), K(aBound.ub(1));
    Value zero = std_constant_index(0);

    // Initialize alloc/R to zero.
    ValueRange zeroLoop = krnl_define_loop(2);
    krnl_iterate(zeroLoop, {zero, zero}, {I, J}, {}, [&](ValueRange args) {
      ValueRange indices = krnl_get_induction_var_value(zeroLoop);
      krnl_store(zeroVal, R, indices);
    });

    // Prepare for the computations.
    // 1) Define blocking, with simdization along the j axis.
    const int64_t iCacheTile(64), jCacheTile(128), kCacheTile(128);
    const int64_t iRegTile(4), jRegTile(8);
    // 2) Alloc data for tiles.
    MemRefType aTileType =
        MemRefType::get({iCacheTile, kCacheTile}, elementType);
    MemRefType bTileType =
        MemRefType::get({kCacheTile, jCacheTile}, elementType);
    // Because the sizes are compile time, no need to pass a dynamic indices.
    SmallVector<IndexExpr> empty;
    Value aBuff =
        insertAllocAndDeallocSimple(rewriter, gemmOp, aTileType, loc, empty);
    Value bBuff =
        insertAllocAndDeallocSimple(rewriter, gemmOp, bTileType, loc, empty);
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
    krnl_permute({ii1, ii2, ii3, jj1, jj2, jj3, kk1, kk2},
        {/*i*/ 2, 4, 5, /*j*/ 0, 3, 6, /*k*/ 1, 7});

    // Compute: A[i, k] * b[k, j] -> R[i, j])
    krnl_iterate(
        {jj, kk}, {jj1, kk1}, {zero, zero}, {J, K}, {}, [&](ValueRange args) {
          ValueRange j1_k1_indices = krnl_get_induction_var_value({jj1, kk1});
          Value j1(j1_k1_indices[0]), k1(j1_k1_indices[1]);
          krnl_copy_to_buffer(bBuff, B, {k1, j1}, zeroVal);
          krnl_iterate({ii}, {ii1}, {zero}, {I}, {}, [&](ValueRange args) {
            ValueRange i1_index = krnl_get_induction_var_value({ii1});
            Value i1(i1_index[0]);
            krnl_copy_to_buffer(aBuff, A, {i1, k1}, zeroVal);
            krnl_iterate(
                {}, {jj2, ii2}, (ValueRange){}, {}, {}, [&](ValueRange args) {
                  ValueRange j2_i2_indices =
                      krnl_get_induction_var_value({jj2, ii2});
                  Value j2(j2_i2_indices[0]), i2(j2_i2_indices[1]);
                  krnl_matmul(aBuff, {i1, k1}, bBuff, {k1, j1}, R, {zero, zero},
                      /*loops*/ {ii3, jj3, kk2}, /*compute start*/ {i2, j2, k1},
                      /*ubs*/ {I, J, K},
                      /*compute tile*/ {iRegTile, jRegTile, kCacheTile},
                      /* a/b/c tiles*/ {}, {}, {}, true, true, false);
                });
          });
        });

    // Define blocked loop and permute.
    ValueRange outerLoops = krnl_define_loop(2);
    krnl_iterate(
        outerLoops, rBound.getLbs(), rBound.getUbs(), {}, [&](ValueRange args) {
          // Outer loop indices.
          ValueRange outerIndices = krnl_get_induction_var_value(outerLoops);

          // Handle alpha/beta coefficients.
          Value res = std_mulf(alphaVal, krnl_load(R, outerIndices));
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

  void genericGemm(ONNXGemmOp &gemmOp, ONNXGemmOpAdaptor &operandAdaptor,
      Type elementType, ONNXGemmOpShapeHelper &shapeHelper, Value alloc,
      Value zeroVal, Value alphaVal, Value betaVal,
      ConversionPatternRewriter &rewriter, Location loc) const {
    // Loop iterations N=0 & M-1 going over each of the res[n, m] values.
    BuildKrnlLoop outputLoops(rewriter, loc, 2);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // Compute the access functions for res[n,m].
    DimIndexExpr n(outputLoops.getInductionVar(0));
    DimIndexExpr m(outputLoops.getInductionVar(1));
    SmallVector<IndexExpr, 4> resAccessFct({n, m});

    // Insert res[n,m] = 0.
    // Create a local reduction value for res[n,m].
    Value reductionVal =
        rewriter.create<AllocaOp>(loc, MemRefType::get({}, elementType));
    rewriter.create<KrnlStoreOp>(loc, zeroVal, reductionVal, ArrayRef<Value>{});

    // Create the inner reduction loop.
    BuildKrnlLoop innerLoops(rewriter, loc, 1);
    innerLoops.createDefineOp();
    innerLoops.pushBounds(0, shapeHelper.aDims[1]);
    innerLoops.createIterateOp();

    // Now start writing code inside the inner loop: get A & B access
    // functions.
    auto ipOuterLoopRegion = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(innerLoops.getIterateBlock());
    {
      DimIndexExpr k(innerLoops.getInductionVar(0));
      SmallVector<IndexExpr, 4> aAccessFct, bAccessFct;
      if (gemmOp.transA() != 0)
        aAccessFct = {k, n};
      else
        aAccessFct = {n, k};
      if (gemmOp.transB() != 0)
        bAccessFct = {m, k};
      else
        bAccessFct = {k, m};
      // Add mat mul operation.
      Value loadedA = krnl_load(operandAdaptor.A(), aAccessFct);
      Value loadedB = krnl_load(operandAdaptor.B(), bAccessFct);
      Value loadedY =
          rewriter.create<KrnlLoadOp>(loc, reductionVal, ArrayRef<Value>{});
      Value AB = rewriter.create<MulFOp>(loc, loadedA, loadedB);
      Value accumulated = rewriter.create<AddFOp>(loc, loadedY, AB);
      rewriter.create<KrnlStoreOp>(
          loc, accumulated, reductionVal, ArrayRef<Value>{});
    }
    rewriter.restoreInsertionPoint(ipOuterLoopRegion);
    Value loadedAB =
        rewriter.create<KrnlLoadOp>(loc, reductionVal, ArrayRef<Value>{});

    // Write code after the completion of the inner loop.
    // Compute the c access function using the broadcast rules.
    SmallVector<IndexExpr, 4> cAccessFct;
    if (shapeHelper.hasBias) {
      for (int x = 2 - shapeHelper.cRank; x < 2; ++x) {
        // If dim > 1, use loop index, otherwise broadcast on 0's element.
        SymbolIndexExpr dim(shapeHelper.cDims[x]);
        cAccessFct.emplace_back(IndexExpr::select(dim > 1, resAccessFct[x], 0));
      }
    }

    // Calculate reduction(AB)*alpha.
    Value alphaAB = rewriter.create<MulFOp>(loc, alphaVal, loadedAB);
    if (shapeHelper.hasBias) {
      // Res = AB*alpha + beta * C.
      Value loadedC = krnl_load(operandAdaptor.C(), cAccessFct);
      auto betaC = rewriter.create<MulFOp>(loc, betaVal, loadedC);
      auto Y = rewriter.create<AddFOp>(loc, alphaAB, betaC);
      krnl_store(Y, alloc, resAccessFct);
    } else {
      // No bias, just store alphaAB into res.
      krnl_store(alphaAB, alloc, resAccessFct);
    }
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Get shape.
    ONNXGemmOpAdaptor operandAdaptor(operands);
    ONNXGemmOp gemmOp = llvm::cast<ONNXGemmOp>(op);
    Location loc = op->getLoc();
    ONNXGemmOpShapeHelper shapeHelper(&gemmOp, &rewriter);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));
    // Scope for krnl EDSC ops
    using namespace mlir::edsc;
    ScopedContext scope(rewriter, loc);
    IndexExprScope outerScope;

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

    // AEE DEGUG: add "true ||" to force the simple nonoptimized, working solution
    if (gemmOp.transA() || gemmOp.transB()) {
      genericGemmV2(gemmOp, operandAdaptor, elementType, shapeHelper, alloc,
          zero, alpha, beta, rewriter, loc);
    } else {
      // AEE transpose not yet implemented
      tiledTransposedGemm(gemmOp, operandAdaptor, elementType, shapeHelper,
          alloc, zero, alpha, beta, rewriter, loc);
    }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXGemmOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpLowering<ONNXGemmOp>>(ctx);
}
