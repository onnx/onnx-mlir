/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Matmul.cpp - Lowering Matmul Op --------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "matmul"
static constexpr int32_t DISABLE_MAT_VEC_PRODUCT = 0;

using namespace mlir;

namespace onnx_mlir {

struct ONNXMatMulOpLowering : public OpConversionPattern<ONNXMatMulOp> {
  ONNXMatMulOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      DimAnalysis *dimAnalysis, bool enableTiling, bool enableSIMD,
      bool enableParallel)
      : OpConversionPattern(typeConverter, ctx), dimAnalysis(dimAnalysis),
        enableTiling(enableTiling), enableSIMD(enableSIMD) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXMatMulOp::getOperationName());
  }

  DimAnalysis *dimAnalysis;
  bool enableTiling;
  bool enableSIMD;
  bool enableParallel;

  // Handle the generic cases, including when there are broadcasts.
  void replaceGenericMatmul(Operation *op, ONNXMatMulOpAdaptor &operandAdaptor,
      Type elementType, ONNXMatMulOpShapeHelper &shapeHelper, Value alloc,
      Value fZero, ConversionPatternRewriter &rewriter, Location loc,
      bool enableParallel) const {

    onnxToKrnlSimdReport(op, /*successful*/ false, /*vl*/ 0, /*trip count*/ 0,
        "no simd for generic algo");

    // Define loops and bounds.
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder> create(rewriter, loc);
    int outerLoopNum = shapeHelper.getOutputDims().size();
    int totLoopNum = outerLoopNum + 1; // Add reduction inner loop.
    ValueRange loopDef = create.krnl.defineLoops(totLoopNum);
    SmallVector<IndexExpr, 4> loopLbs(totLoopNum, LitIE(0));
    SmallVector<IndexExpr, 4> loopUbs; // All getOutputDims, plus reduction.
    SmallVector<Value, 4> outerLoops;  // All but the last loop def.
    for (int i = 0; i < outerLoopNum; ++i) {
      loopUbs.emplace_back(shapeHelper.getOutputDims()[i]);
      outerLoops.emplace_back(loopDef[i]);
    }
    int aRank = shapeHelper.aDims.size();
    int bRank = aRank; // Add for better readability.
    IndexExpr innerUb = shapeHelper.aDims[aRank - 1];
    loopUbs.emplace_back(innerUb);
    SmallVector<Value, 1> innerLoop{loopDef[totLoopNum - 1]}; // Last loop def.
    if (enableParallel) {
      int64_t parId;
      if (findSuitableParallelDimension(loopLbs, loopUbs, 0, 1, parId,
              /*min iter for going parallel*/ 16)) {
        create.krnl.parallel(outerLoops[0]);
        onnxToKrnlParallelReport(
            op, true, 0, loopLbs[0], loopUbs[0], "matmul generic");
      } else {
        onnxToKrnlParallelReport(op, false, 0, loopLbs[0], loopUbs[0],
            "not enough work for matmul generic");
      }
    }

    // Non-reduction loop iterations: output-rank.
    create.krnl.iterateIE(loopDef, outerLoops, loopLbs, loopUbs,
        [&](const KrnlBuilder &createKrnl, ValueRange outerIndices) {
          MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
              createKrnl);

          ValueRange inits = ValueRange(fZero);
          // Inner loop for reduction.
          auto innerIterate = create.krnl.iterate({}, innerLoop, {}, {}, inits,
              [&](const KrnlBuilder &createKrnl, ValueRange innerIndex,
                  ValueRange iterArgs) {
                // Get last argument for the iterate body.
                Value iterArg = iterArgs.back();

                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);
                Value k = innerIndex[0];
                SmallVector<Value, 4> aAccessFct, bAccessFct;
                for (int i = 0; i < aRank; ++i) {
                  // Add index if dim is not a padded dimension.
                  if (!shapeHelper.aPadDims[i]) {
                    // For A, reduction index is last
                    if (i == aRank - 1) {
                      aAccessFct.emplace_back(k);
                    } else {
                      aAccessFct.emplace_back(outerIndices[i]);
                    }
                  }
                  if (!shapeHelper.bPadDims[i]) {
                    // For B, reduction index is second to last.
                    if (i == bRank - 2) {
                      bAccessFct.emplace_back(k);
                    } else if (i == outerLoopNum) {
                      // When the rank of A 1D, then the output lost one
                      // dimension. E,g, (5) x (10, 5, 4) -> padded (1, 5) x
                      // (10, 5, 4) = (10, 1, 4). But we drop the "1" so its
                      // really (10, 4). When processing the last dim of the
                      // reduction (i=2 here), we would normally access
                      // output[2] but it does not exist, because we lost a dim
                      // in the output due to 1D A.
                      bAccessFct.emplace_back(outerIndices[i - 1]);
                    } else {
                      bAccessFct.emplace_back(outerIndices[i]);
                    }
                  }
                }
                // Add mat mul operation.
                Value loadedA =
                    create.krnl.load(operandAdaptor.getA(), aAccessFct);
                Value loadedB =
                    create.krnl.load(operandAdaptor.getB(), bAccessFct);
                Value loadedY = iterArg;
                Value AB = create.math.mul(loadedA, loadedB);
                Value accumulated = create.math.add(loadedY, AB);
                // Create yield.
                create.krnl.yield(accumulated);
              });
          Value accumulated = innerIterate.getResult(0);
          create.krnl.store(accumulated, alloc, outerIndices);
          // Create yield.
          create.krnl.yield({});
        });
  }

  void computeTileSizeForMatMatProduct(Operation *op, DimIndexExpr dimI,
      DimIndexExpr dimJ, DimIndexExpr dimK, int64_t &iRegTile,
      int64_t &jRegTile, int64_t &kRegTile, bool &simdize) const {

    // Default values
    iRegTile = 4;
    jRegTile = 8; // SIMD dim.
    kRegTile = 8;

    if (!simdize)
      onnxToKrnlSimdReport(op, /*successful*/ false, /*vl*/ 0, /*trip count*/ 0,
          "no simd because disabled for mat * mat");

    if (dimI.isLiteral()) {
      int64_t constI = dimI.getLiteral();
      if (constI < iRegTile) {
        iRegTile = constI;
        LLVM_DEBUG({
          llvm::dbgs() << "MatMul: Tiling I is reduced to " << iRegTile << "\n";
        });
      }
    }

    if (dimJ.isLiteral()) {
      int64_t constJ = dimJ.getLiteral();
      // No tiling needed when J dim is 1.
      if (constJ == 1) {
        // no tiling needed
        jRegTile = 1;
        LLVM_DEBUG({
          llvm::dbgs() << "MatMul: Tiling J is set to " << jRegTile << "\n";
        });

        // When jRegTile does not divide J, but 4 would, use 4, unless J is very
        // large, in which case it is better to simdize well the steady state
        // and ignore the last partial block.
      } else if (constJ % jRegTile != 0 && constJ % 4 == 0 && constJ <= 32) {
        jRegTile = 4;
        LLVM_DEBUG({
          llvm::dbgs() << "MatMul: Tiling J is reduced to " << jRegTile << "\n";
        });
      }
      // Simdization occurs along j and jRegTile. If dimJ is smaller than
      // jRegTile, disable simdization.
      if (constJ < jRegTile) {
        simdize = false;
        onnxToKrnlSimdReport(op, /*successful*/ false, /*vl*/ 0,
            /*trip count*/ 0, "no simd in mat * mat because j-dim too small");
        LLVM_DEBUG({
          llvm::dbgs() << "MatMul: Disable simdization because trip " << constJ
                       << " is smaller than reg tile " << jRegTile << "\n";
        });
      }
    }

    if (dimK.isLiteral()) {
      int64_t constK = dimK.getLiteral();
      if (constK < kRegTile) {
        kRegTile = constK;
        LLVM_DEBUG({
          llvm::dbgs() << "MatMul: Tiling K is reduced to " << kRegTile << "\n";
        });
      }
    }

    if (simdize)
      onnxToKrnlSimdReport(op, /*successful*/ true, /*vl*/ jRegTile,
          /*trip count*/ jRegTile, "simd for mat * mat along j dim");
    LLVM_DEBUG({
      llvm::dbgs() << "MatMul mat: Tiling I " << iRegTile << ", J " << jRegTile
                   << ", K " << kRegTile << ", simd " << simdize << "\n";
    });
  }

  void computeTileSizeForMatVectProduct(Operation *op, int64_t VL,
      DimIndexExpr dimI, DimIndexExpr dimJ, DimIndexExpr dimK,
      int64_t &iRegTile, int64_t &jRegTile, int64_t &kRegTile,
      bool &simdize) const {

    if (!simdize)
      onnxToKrnlSimdReport(op, /*successful*/ false, /*vl*/ 0, /*trip count*/ 0,
          "no simd because disabled for mat * vec");

    // Default values.
    // Right can only tile i and k by (possibly distinct) multiple of VL.
    iRegTile = 2 * VL; // SIMD dim during multi-reduction.
    jRegTile = 1;
    kRegTile = 16 * VL; // SIMD dim during multiplication.

    if (dimK.isLiteral()) {
      int64_t constK = dimK.getLiteral();
      // Register tile in the I Dim is really for the reduction. The
      // computations will be further tiled to a multiple of VL inside
      // krnl.matmul.
      kRegTile = (constK / VL) * VL; // largest multiple
      if (kRegTile > 64 * VL) {
        kRegTile = 64 * VL;
        LLVM_DEBUG({ llvm::dbgs() << "MatMul Vec: cap tiling k\n"; });
      } else if (kRegTile < VL) {
        // Not enough data, can only support i/k reg tile of 4.
        LLVM_DEBUG({ llvm::dbgs() << "MatMul Vec: disable k\n"; });
        simdize = false;
        kRegTile = 1;
        onnxToKrnlSimdReport(op, /*successful*/ false, /*vl*/ 0,
            /*trip count*/ 0, "no simd in mat * vec as k-dim too small");
      }
    }
    if (dimI.isLiteral()) {
      int64_t constI = dimI.getLiteral();
      if (constI < iRegTile) {
        iRegTile = (constI / VL) * VL; // largest multiple
        if (iRegTile < VL) {
          // Not enough data, can only support i/k reg tile of 4.
          LLVM_DEBUG({ llvm::dbgs() << "MatMul Vec: disable i\n"; });
          simdize = false;
          iRegTile = 1;
          onnxToKrnlSimdReport(op, /*successful*/ false, /*vl*/ 0,
              /*trip count*/ 0, "no simd in mat * vec because i-dim too small");
        }
      }
    }

    if (simdize)
      onnxToKrnlSimdReport(op, /*successful*/ true, /*vl*/ kRegTile,
          /*trip count*/ kRegTile,
          "simd for mat * vec along k dim (shuffle on i dim)");
    LLVM_DEBUG({
      llvm::dbgs() << "MatMul vec: Tiling I " << iRegTile << ", J " << jRegTile
                   << ", K " << kRegTile << ", simd " << simdize << "\n";
    });
  }

  // Handle the cases with 2x2 matrices both for A, B, and C without
  // broadcast. Implementation here uses the efficient 1d tiling plus kernel
  // substitution.
  void replace2x2Matmul2d(Operation *op, ONNXMatMulOpAdaptor &operandAdaptor,
      Type elementType, ONNXMatMulOpShapeHelper &shapeHelper, Value alloc,
      Value zeroVal, ConversionPatternRewriter &rewriter, Location loc,
      bool enableParallel) const {
    // Prepare: loop bounds and zero
    Value A(operandAdaptor.getA()), B(operandAdaptor.getB()), C(alloc);
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder, VectorBuilder>
        create(rewriter, loc);
    Value zero = create.math.constantIndex(0);
    Value I = create.mem.dim(C, 0);
    Value J = create.mem.dim(C, 1);
    Value K = create.mem.dim(A, 1);

    // Initialize alloc/C to zero.
    create.krnl.memset(alloc, zeroVal);
    bool simdize = enableSIMD;

    // Define blocking, with simdization along the j axis.
    DimIndexExpr dimI(I), dimJ(J), dimK(K);
    LiteralIndexExpr zeroIE(0);
    int64_t iRegTile, jRegTile, kRegTile;
    bool isMatVectorProduct =
        !DISABLE_MAT_VEC_PRODUCT && dimJ.isLiteral() && dimJ.getLiteral() == 1;
    if (isMatVectorProduct) {
      int64_t archVL = create.vec.getArchVectorLength(elementType);
      computeTileSizeForMatVectProduct(
          op, archVL, dimI, dimJ, dimK, iRegTile, jRegTile, kRegTile, simdize);
    } else {
      computeTileSizeForMatMatProduct(
          op, dimI, dimJ, dimK, iRegTile, jRegTile, kRegTile, simdize);
    }

    // I, J, K loop.
    ValueRange origLoop = create.krnl.defineLoops(3);
    Value ii(origLoop[0]), jj(origLoop[1]), kk(origLoop[2]);
    // Define blocked loop and permute.
    ValueRange iRegBlock = create.krnl.block(ii, iRegTile);
    Value ii1(iRegBlock[0]), ii2(iRegBlock[1]);
    ValueRange jRegBlock = create.krnl.block(jj, jRegTile);
    Value jj1(jRegBlock[0]), jj2(jRegBlock[1]);
    ValueRange kRegBlock = create.krnl.block(kk, kRegTile);
    Value kk1(kRegBlock[0]), kk2(kRegBlock[1]);
    create.krnl.permute({ii1, ii2, jj1, jj2, kk1, kk2}, {0, 3, 1, 4, 2, 5});
    if (enableParallel) {
      int64_t parId;
      SmallVector<IndexExpr, 1> lb(1, zeroIE), ub(1, dimI);
      if (findSuitableParallelDimension(lb, ub, 0, 1, parId,
              /*min iter for going parallel*/ 4 * iRegTile)) {
        create.krnl.parallel(ii1);
        onnxToKrnlParallelReport(
            op, true, 0, zeroIE, dimI, "matmul no broadcast");
      } else {
        onnxToKrnlParallelReport(op, false, 0, zeroIE, dimI,
            "not enough work for matmul no broadcast");
      }
    }
    create.krnl.iterate({ii, jj, kk}, {ii1, jj1, kk1}, {zero, zero, zero},
        {I, J, K}, [&](const KrnlBuilder &createKrnl, ValueRange indices) {
          Value i1(indices[0]), j1(indices[1]), k1(indices[2]);
          createKrnl.matmul(A, {zero, zero}, B, {zero, zero}, C, {zero, zero},
              {ii2, jj2, kk2}, {i1, j1, k1}, {I, J, K},
              {iRegTile, jRegTile, kRegTile}, {}, {}, {}, simdize,
              /*unroll*/ true, /*overCompute*/ false);
        });
  }

  // Handle the cases with 2x2 matrices with broadcasting.
  // Either broadcast A (and then B has rank 2, broadcastingB is false) or
  // broadcast B (and then A has rank 2, broadcastingB is true). But we can also
  // use this same algorithm when both A and B have identical, static
  // broadcasting ranks. In such case, sameStaticBroadcast is true, and the
  // value of broadcastingB does not matter as they treated as both
  // broadcasting.
  void replace2x2Matmul2dBroadcasting(Operation *op,
      ONNXMatMulOpAdaptor &operandAdaptor, Type elementType,
      ONNXMatMulOpShapeHelper &shapeHelper, bool broadcastingB,
      bool sameStaticBroadcast, Value alloc, Value zeroVal,
      ConversionPatternRewriter &rewriter, Location loc,
      bool enableParallel) const {
    // Prepare: loop bounds and zero
    Value A(operandAdaptor.getA()), B(operandAdaptor.getB()), C(alloc);
    int64_t ARank = shapeHelper.aDims.size();
    int64_t BRank = shapeHelper.bDims.size();
    int64_t broadcastRank = (broadcastingB ? BRank : ARank) - 2;
    assert(broadcastRank > 0 && "expected broadcast dims for A or B");
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder, VectorBuilder>
        create(rewriter, loc);
    Value zero = create.math.constantIndex(0);
    Value I = create.mem.dim(C, broadcastRank + 0); // C has broadcast.
    Value J = create.mem.dim(C, broadcastRank + 1); // C has broadcast.
    // When sameStaticBroadcast: A & B have broadcast, take K from broadcastRank
    // +0 of B (same as +1 from A). When broadcasting B, A has no broadcast,
    // take K from 2nd dim. When not broadcasting B, B has no broadcast, take K
    // from 1st dim.
    Value K = sameStaticBroadcast ? create.mem.dim(B, broadcastRank + 0)
                                  : (broadcastingB ? create.mem.dim(A, 1)
                                                   : create.mem.dim(B, 0));

    // Initialize alloc/C to zero.
    create.krnl.memset(alloc, zeroVal);
    bool simdize = enableSIMD;

    // Define blocking, with simdization along the j axis.
    DimIndexExpr dimI(I), dimJ(J), dimK(K);
    int64_t iRegTile, jRegTile, kRegTile;
    bool isMatVectorProduct =
        !DISABLE_MAT_VEC_PRODUCT && dimJ.isLiteral() && dimJ.getLiteral() == 1;
    if (isMatVectorProduct) {
      int64_t archVL = create.vec.getArchVectorLength(elementType);
      computeTileSizeForMatVectProduct(
          op, archVL, dimI, dimJ, dimK, iRegTile, jRegTile, kRegTile, simdize);
    } else {
      computeTileSizeForMatMatProduct(
          op, dimI, dimJ, dimK, iRegTile, jRegTile, kRegTile, simdize);
    }

    // Broadcast loops
    ValueRange broadcastLoop = create.krnl.defineLoops(broadcastRank);
    SmallVector<Value, 4> broadcastLB(broadcastRank, zero);
    SmallVector<Value, 4> broadcastUB;
    for (int64_t i = 0; i < broadcastRank; ++i)
      broadcastUB.emplace_back(create.mem.dim(C, i));
    if (enableParallel) {
      int64_t parId;
      // Could check out more than the outer dim of the broadcasts...
      SmallVector<IndexExpr, 1> lb(1, LitIE(0)),
          ub(1, shapeHelper.getOutputDims()[0]);
      if (findSuitableParallelDimension(lb, ub, 0, 1, parId,
              /*min iter for going parallel*/ 4)) {
        create.krnl.parallel(broadcastLoop[0]);
        onnxToKrnlParallelReport(op, true, 0, lb[0], ub[0], "matmul broadcast");
      } else {
        onnxToKrnlParallelReport(
            op, false, 0, lb[0], ub[0], "not enough work in matmul broadcast");
      }
    }
    create.krnl.iterate(broadcastLoop, broadcastLoop, broadcastLB, broadcastUB,
        [&](const KrnlBuilder &createKrnl, ValueRange broadcastIndices) {
          MultiDialectBuilder<KrnlBuilder> create(createKrnl);
          // I, J, K loop.
          ValueRange origLoop = create.krnl.defineLoops(3);
          // IJK indices.
          Value ii(origLoop[0]), jj(origLoop[1]), kk(origLoop[2]);
          // Define blocked loop and permute.
          ValueRange iRegBlock = create.krnl.block(ii, iRegTile);
          Value ii1(iRegBlock[0]), ii2(iRegBlock[1]);
          ValueRange jRegBlock = create.krnl.block(jj, jRegTile);
          Value jj1(jRegBlock[0]), jj2(jRegBlock[1]);
          ValueRange kRegBlock = create.krnl.block(kk, kRegTile);
          Value kk1(kRegBlock[0]), kk2(kRegBlock[1]);
          create.krnl.permute(
              {ii1, ii2, jj1, jj2, kk1, kk2}, {0, 3, 1, 4, 2, 5});
          create.krnl.iterate({ii, jj, kk}, {ii1, jj1, kk1}, {zero, zero, zero},
              {I, J, K},
              [&](const KrnlBuilder &createKrnl, ValueRange indices) {
                Value i1(indices[0]), j1(indices[1]), k1(indices[2]);
                // Compute global start for B/C: {broadcastIndices, 0, 0}
                SmallVector<Value, 4> broadcastGlobalStart;
                for (int64_t i = 0; i < broadcastRank; ++i)
                  broadcastGlobalStart.emplace_back(broadcastIndices[i]);
                broadcastGlobalStart.emplace_back(zero);
                broadcastGlobalStart.emplace_back(zero);
                if (sameStaticBroadcast) {
                  // Each of A, B, & C starts at broadcastGlobalStart.
                  createKrnl.matmul(A, broadcastGlobalStart, B,
                      broadcastGlobalStart, C, broadcastGlobalStart,
                      {ii2, jj2, kk2}, {i1, j1, k1}, {I, J, K},
                      {iRegTile, jRegTile, kRegTile}, {}, {}, {}, simdize,
                      /*unroll*/ true, /*overCompute*/ false);
                } else if (broadcastingB) {
                  // B & C start at broadcastGlobalStart, A starts at {0,0}.
                  createKrnl.matmul(A, {zero, zero}, B, broadcastGlobalStart, C,
                      broadcastGlobalStart, {ii2, jj2, kk2}, {i1, j1, k1},
                      {I, J, K}, {iRegTile, jRegTile, kRegTile}, {}, {}, {},
                      simdize, /*unroll*/ true, /*overCompute*/ false);
                } else {
                  // A & C start at broadcastGlobalStart, B starts at {0,0}.
                  createKrnl.matmul(A, broadcastGlobalStart, B, {zero, zero}, C,
                      broadcastGlobalStart, {ii2, jj2, kk2}, {i1, j1, k1},
                      {I, J, K}, {iRegTile, jRegTile, kRegTile}, {}, {}, {},
                      simdize, /*unroll*/ true, /*overCompute*/ false);
                }
              });
        });
  }

  // Handle the cases with 2x2 matrices both for A, B, and C without
  // broadcast, broadcast of A to rank 2 B,  broadcast of B to rank 2 A, or
  // static, identical shaped broadcasting size A & B.
  // Implementation here uses the efficient 2d tiling plus kernel substitution.
  LogicalResult matchAndRewrite(ONNXMatMulOp matMulOp,
      ONNXMatMulOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = matMulOp.getOperation();
    ValueRange operands = adaptor.getOperands();
    Location loc = ONNXLoc<ONNXMatMulOp>(op);
    MultiDialectBuilder<IndexExprBuilderForKrnl, MathBuilder, MemRefBuilder>
        create(rewriter, loc);
    // Get shape.
    ONNXMatMulOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Insert an allocation and deallocation for the output of this operation.
    Type elementType = outputMemRefType.getElementType();
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Get the constants: zero.
    Value zero = create.math.constant(elementType, 0);

    Value A(adaptor.getA()), B(adaptor.getB());
    int aRank = mlir::cast<MemRefType>(A.getType()).getShape().size();
    int bRank = mlir::cast<MemRefType>(B.getType()).getShape().size();
    int cRank = mlir::cast<MemRefType>(alloc.getType()).getShape().size();
    if (enableTiling && aRank == 2 && bRank == 2) {
      // Optimized Matmul only when 2D and allowed to tile and unroll.
      assert(cRank == 2 && "expected IxK * KxJ = IxJ 2D result");
      replace2x2Matmul2d(op, adaptor, elementType, shapeHelper, alloc, zero,
          rewriter, loc, enableParallel);
    } else if (enableTiling && aRank == 2 && bRank > 2) {
      // Broadcasting B.
      assert(cRank == bRank && "expected IxK * *xKxJ = *xIxJ result");
      replace2x2Matmul2dBroadcasting(op, adaptor, elementType, shapeHelper,
          /*broadcasting B*/ true,
          /*same static broadcast*/ false, alloc, zero, rewriter, loc,
          enableParallel);
    } else if (enableTiling && aRank > 2 && bRank == 2) {
      // Broadcasting A.
      assert(cRank == aRank && "expected IxK * *xKxJ = *xIxJ result");
      replace2x2Matmul2dBroadcasting(op, adaptor, elementType, shapeHelper,
          /*broadcasting B*/ false,
          /*same static broadcast*/ false, alloc, zero, rewriter, loc,
          enableParallel);
    } else {
      // Test if have A and B have identical batch size.
      bool sameBatchSize = (enableTiling && aRank > 2 && aRank == bRank);
      if (sameBatchSize) {
        for (int i = 0; i < aRank - 2; ++i)
          // Note that using A and B from the operation instead of adaptor.
          // It's because DimAnalysis has been done on operations.
          if (!dimAnalysis->sameDim(matMulOp.getA(), i, matMulOp.getB(), i)) {
            sameBatchSize = false;
            break;
          }
      }
      // While there is technically no broadcasting there, we can use nearly the
      // same logic as in replace2x2Matmul2dBroadcasting. So reuse that code.
      if (sameBatchSize) {
        assert(cRank == aRank && "expected IxK * *xKxJ = *xIxJ result");
        replace2x2Matmul2dBroadcasting(op, adaptor, elementType, shapeHelper,
            /*broadcasting B*/ true,
            /*same static broadcast*/ true, alloc, zero, rewriter, loc,
            enableParallel);
      } else {
        replaceGenericMatmul(op, adaptor, elementType, shapeHelper, alloc, zero,
            rewriter, loc, enableParallel);
      }
    }
    // Done.
    rewriter.replaceOp(op, alloc);
    return success();
  }
}; // namespace onnx_mlir

void populateLoweringONNXMatMulOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, DimAnalysis *dimAnalysis,
    bool enableTiling, bool enableSIMD, bool enableParallel) {
  patterns.insert<ONNXMatMulOpLowering>(typeConverter, ctx, dimAnalysis,
      enableTiling, enableSIMD, enableParallel);
}

} // namespace onnx_mlir
