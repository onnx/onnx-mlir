/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Gemm.cpp - Lowering Gemm Op ------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
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
    if (enableParallel) {
      int64_t parId;
      if (findSuitableParallelDimension(loopLbs, loopUbs, 0, 1, parId,
              /*min iter for going parallel*/ 4)) {
        create.krnl.parallel(outerLoopDef[0]);
        onnxToKrnlParallelReport(op, true, parId, loopLbs[parId],
            loopUbs[parId], "generic GEMM on outer loop");
      } else {
        onnxToKrnlParallelReport(op, false, parId, loopLbs[parId],
            loopUbs[parId], "not enough work for parallel generic GEMM");
      }
    }
    create.krnl.iterateIE(loopDef, outerLoopDef, loopLbs, loopUbs,
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
      if (enableParallel) {
        int64_t parId;
        SmallVector<IndexExpr, 1> lb(1, zeroIE), ub(1, I);
        if (findSuitableParallelDimension(lb, ub, 0, 1, parId,
                /*min iter for going parallel*/ 4 * iCacheTile)) {
          create.krnl.parallel(ii1);
          onnxToKrnlParallelReport(
              op, true, 0, zeroIE, I, "GEMM tiled copy I parallel");
        } else {
          onnxToKrnlParallelReport(op, false, 0, zeroIE, I,
              "not enough work for GEMM tiled copy I parallel");
        }
      }
      // Compute: A[i, k] * b[k, j] -> R[i, j])
      create.krnl.iterateIE({ii, jj, kk}, {ii1, jj1}, {zeroIE, zeroIE, zeroIE},
          {I, J, K},
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
      if (enableParallel) {
        int64_t parId;
        SmallVector<IndexExpr, 1> lb(1, zeroIE), ub(1, J);
        if (findSuitableParallelDimension(lb, ub, 0, 1, parId,
                /*min iter for going parallel*/ 4 * jCacheTile)) {
          create.krnl.parallel(jj1);
          onnxToKrnlParallelReport(
              op, true, 0, zeroIE, J, "GEMM tiled no copy J parallel");
        } else {
          onnxToKrnlParallelReport(op, false, 0, zeroIE, J,
              "not enough work for GEMM tiled no copy J parallel");
        }
      }
      // Compute: A[i, k] * b[k, j] -> R[i, j])
      // Krnl Rule: must put all the iter bounds at once, but can only put the
      // "not currently used ones" like ii here last. Gave an error when ii was
      // listed first.
      create.krnl.iterateIE({jj, kk, ii}, {jj1, kk1}, {zeroIE, zeroIE, zeroIE},
          {J, K, I},
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
    if (enableParallel) {
      int64_t parId;
      SmallVector<IndexExpr, 1> lb(1, zeroIE), ub(1, I);
      if (findSuitableParallelDimension(lb, ub, 0, 1, parId,
              /*min iter for going parallel*/ 16)) {
        create.krnl.parallel(outerLoops[0]);
        onnxToKrnlParallelReport(
            op, true, 0, zeroIE, I, "outer loop on tiled Transposed Gemm");
      } else {
        onnxToKrnlParallelReport(op, false, 0, zeroIE, I,
            "not enough work for outer loop on tiled Transposed Gemm");
      }
    }
    create.krnl.iterateIE(outerLoops, outerLoops, {zeroIE, zeroIE}, {I, J},
        [&](const KrnlBuilder &createKrnl, ValueRange outerIndices) {
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

    if (enableTiling && !DEBUG_OPTIMIZED_OFF) {
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
