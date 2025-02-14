/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- KrnlMatmul.cpp - Lower KrnlMatmulOp -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlMatmulOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Conversion/KrnlToAffine/KrnlToAffineHelper.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Support/KrnlSupport.hpp"
#include "llvm/Support/Debug.h"

#include <mutex>

#define DEBUG_TYPE "krnl_to_affine"

static constexpr int32_t DISABLE_MAT_VEC_PRODUCT = 0;

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

static constexpr int BUFFER_ALIGN = 64;
extern UnrollAndJamMap unrollAndJamMap;
extern std::mutex unrollAndJamMutex;

// KrnlMatmul will be lowered to vector and affine expressions
class KrnlMatmulLowering : public ConversionPattern {
public:
  explicit KrnlMatmulLowering(
      TypeConverter &typeConverter, MLIRContext *context, bool parallelEnabled)
      : ConversionPattern(
            typeConverter, KrnlMatMulOp::getOperationName(), 1, context) {
    this->parallelEnabled = parallelEnabled;
  }

  bool parallelEnabled = false;

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto matmulOp = mlir::cast<KrnlMatMulOp>(op);
    KrnlMatMulOpAdaptor operandAdaptor(matmulOp);
    // Option.
    bool fullUnrollAndJam = matmulOp.getUnroll();

    // Operands and types.
    Type elementType = mlir::cast<MemRefType>(operandAdaptor.getA().getType())
                           .getElementType();
    bool simdize = matmulOp.getSimdize();
    // Init scope and emit constants.
    Location loc = matmulOp.getLoc();
    MultiDialectBuilder<AffineBuilderKrnlMem, VectorBuilder,
        IndexExprBuilderForKrnl>
        create(rewriter, loc);
    IndexExprScope indexScope(create.affineKMem);

    // Gather A, B, C tile sizes.
    SmallVector<IndexExpr, 2> aTileSize, bTileSize, cTileSize;
    Value A(operandAdaptor.getA()), B(operandAdaptor.getB()),
        C(operandAdaptor.getC());
    SmallVector<IndexExpr, 4> aBounds, bBounds, cBounds, aTileSizeFromAttr,
        bTileSizeFromAttr, cTileSizeFromAttr, computeTileSizeFromAttr;
    create.krnlIE.getShapeAsSymbols(A, aBounds);
    create.krnlIE.getShapeAsSymbols(B, bBounds);
    create.krnlIE.getShapeAsSymbols(C, cBounds);
    int64_t aRank(aBounds.size()), bRank(bBounds.size()), cRank(cBounds.size());
    // Tile sizes for A/B/C are determined by their memref unless explicitly
    // specified by an optional argument. That allows A/B/C memrefs to be
    // padded if needed for SIMD/unroll and jam, for example.
    create.krnlIE.getIntFromArrayAsLiterals(
        matmulOp.getATileSizeAttr(), aTileSizeFromAttr);
    if (aTileSizeFromAttr.size())
      aTileSize = {aTileSizeFromAttr[0], aTileSizeFromAttr[1]};
    else
      aTileSize = {aBounds[aRank - 2], aBounds[aRank - 1]};
    create.krnlIE.getIntFromArrayAsLiterals(
        matmulOp.getBTileSizeAttr(), bTileSizeFromAttr);
    if (bTileSizeFromAttr.size())
      bTileSize = {bTileSizeFromAttr[0], bTileSizeFromAttr[1]};
    else
      bTileSize = {bBounds[bRank - 2], bBounds[bRank - 1]};
    create.krnlIE.getIntFromArrayAsLiterals(
        matmulOp.getCTileSizeAttr(), cTileSizeFromAttr);
    if (cTileSizeFromAttr.size())
      cTileSize = {cTileSizeFromAttr[0], cTileSizeFromAttr[1]};
    else
      cTileSize = {cBounds[cRank - 2], cBounds[cRank - 1]};

    // Gather N, M, K compute tile size. This is the size of the computations,
    // if the tile is full. Because computation in the buffers could be further
    // sub-tiled, the default size can be overridden from the tile sizes using
    // the computeTileSize attribute. Tiles may not be full if they are at the
    // outer boundaries of the original data.
    IndexExpr iComputeTileSize = cTileSize[0];
    IndexExpr jComputeTileSize = cTileSize[1];
    IndexExpr kComputeTileSize = aTileSize[1];
    create.krnlIE.getIntFromArrayAsLiterals(
        matmulOp.getComputeTileSizeAttr(), computeTileSizeFromAttr);

    if (computeTileSizeFromAttr.size()) {
      iComputeTileSize = computeTileSizeFromAttr[0];
      jComputeTileSize = computeTileSizeFromAttr[1];
      kComputeTileSize = computeTileSizeFromAttr[2];
    }
    // Get the global upper bound of the original computations.
    SymbolIndexExpr iGlobalUB(operandAdaptor.getIGlobalUB()),
        jGlobalUB(operandAdaptor.getJGlobalUB()),
        kGlobalUB(operandAdaptor.getKGlobalUB());

    // Has a matrix times vector when the J upper bound is literal 1.
    bool matVectorProduct = !DISABLE_MAT_VEC_PRODUCT && jGlobalUB.isLiteral() &&
                            jGlobalUB.getLiteral() == 1;

    // Investigate SIMD
    IndexExpr vectorLen = LitIE(1); // Assume no simd.
    if (simdize) {
      if (matVectorProduct) {
        // Matrix (I x K) times vector (K x 1). We currently vectorize along the
        // i (producing VL results at a time), each of which is a reduction
        // along the K-axis.
        if (iComputeTileSize.isLiteral() && kComputeTileSize.isLiteral()) {
          uint64_t i = iComputeTileSize.getLiteral();
          uint64_t k = kComputeTileSize.getLiteral();
          uint64_t archVL = create.vec.getArchVectorLength(elementType);
          if (i % archVL == 0 && k % archVL == 0) {
            // Right now, vector length must be archVL.
            vectorLen = LitIE(archVL);
          } else {
            simdize = false;
            LLVM_DEBUG(llvm::dbgs() << "Matmul: mat*vec with bad sizes: i " << i
                                    << ", k " << k << "\n");
          }
        } else {
          simdize = false;
          LLVM_DEBUG(llvm::dbgs() << "Matmul: mat*vec with non-literal dims\n");
        }
      } else {
        // Matrix times matrix case.
        if (jComputeTileSize.isLiteral()) {
          // We simdize along M for the full compute tile.
          vectorLen = jComputeTileSize;
        } else {
          // Cannot simdize if the vector length is not a compile time constant.
          simdize = false;
          LLVM_DEBUG(
              llvm::dbgs() << "Matmul: No simd due to vl not a literal\n");
        }
      }
    }

    // Now get global start indices, which would define the first element of the
    // tiles in the original computations.
    DimIndexExpr iGlobalIndexComputeStart(
        operandAdaptor.getIGlobalIndexComputeStart()),
        jGlobalIndexComputeStart(operandAdaptor.getJGlobalIndexComputeStart()),
        kGlobalIndexComputeStart(operandAdaptor.getKGlobalIndexComputeStart());
    // A[i, k];
    SmallVector<IndexExpr, 4> aStart, bStart, cStart;
    for (int t = 0; t < aRank - 2; t++)
      aStart.emplace_back(SymIE(operandAdaptor.getAGlobalIndexMemStart()[t]));
    aStart.emplace_back(
        iGlobalIndexComputeStart -
        DimIE(operandAdaptor.getAGlobalIndexMemStart()[aRank - 2]));
    aStart.emplace_back(
        kGlobalIndexComputeStart -
        DimIE(operandAdaptor.getAGlobalIndexMemStart()[aRank - 1]));
    // B[k, j];
    for (int t = 0; t < bRank - 2; t++)
      bStart.emplace_back(SymIE(operandAdaptor.getBGlobalIndexMemStart()[t]));
    bStart.emplace_back(
        kGlobalIndexComputeStart -
        DimIE(operandAdaptor.getBGlobalIndexMemStart()[bRank - 2]));
    bStart.emplace_back(
        jGlobalIndexComputeStart -
        DimIE(operandAdaptor.getBGlobalIndexMemStart()[bRank - 1]));
    // C[i, j]
    for (int t = 0; t < cRank - 2; t++)
      cStart.emplace_back(SymIE(operandAdaptor.getCGlobalIndexMemStart()[t]));
    cStart.emplace_back(
        iGlobalIndexComputeStart -
        DimIE(operandAdaptor.getCGlobalIndexMemStart()[cRank - 2]));
    cStart.emplace_back(
        jGlobalIndexComputeStart -
        DimIE(operandAdaptor.getCGlobalIndexMemStart()[cRank - 1]));

    // Now determine if we have full/partial tiles. This is determined by the
    // outer dimensions of the original computations, as by definition tiling
    // within the buffer always results in full tiles. In other words, partial
    // tiles only occurs because of "running out" of the original data.
    IndexExpr iIsTileFull = create.krnlIE.isTileFull(
        iGlobalIndexComputeStart, iComputeTileSize, iGlobalUB);
    IndexExpr jIsTileFull = create.krnlIE.isTileFull(
        jGlobalIndexComputeStart, jComputeTileSize, jGlobalUB);
    IndexExpr kIsTileFull = create.krnlIE.isTileFull(
        kGlobalIndexComputeStart, kComputeTileSize, kGlobalUB);
    SmallVector<IndexExpr, 3> allFullTiles = {
        iIsTileFull, jIsTileFull, kIsTileFull};

    SmallVector<IndexExpr, 1> jFullTiles = {jIsTileFull};
    // And if the tiles are not full, determine how many elements to compute.
    // With over-compute, this could be relaxed.
    IndexExpr iTrip = create.krnlIE.tileSize(iGlobalIndexComputeStart,
        iComputeTileSize, iGlobalUB); // May or may not be full.
    IndexExpr jTrip = create.krnlIE.tileSize(jGlobalIndexComputeStart,
        jComputeTileSize, jGlobalUB); // May or may not be full.
    IndexExpr kTrip = create.krnlIE.tileSize(kGlobalIndexComputeStart,
        kComputeTileSize, kGlobalUB); // May or may not be full.
    IndexExpr jPartialTrip = create.krnlIE.partialTileSize(
        jGlobalIndexComputeStart, jComputeTileSize, jGlobalUB);

    if (simdize) {
      // SIMD code generator.
      if (matVectorProduct) {
        // Alloc of temp outside of inner if/then/else.
        Value TmpSimdProd = allocForGenSimdMatVect(create.affineKMem,
            elementType, iComputeTileSize, jComputeTileSize, kComputeTileSize,
            vectorLen, fullUnrollAndJam);
        Value TmpScalarProd = allocForGenScalar(create.affineKMem, elementType,
            iTrip, jTrip, kTrip, /*unroll*/ false);
        // clang-format off
        create.affineKMem.ifThenElseIE(indexScope, allFullTiles,
          /* then full tiles */ [&](const AffineBuilderKrnlMem &createAffine) {
          genSimdMatVect(createAffine, matmulOp, TmpSimdProd, elementType, aStart, bStart,
            cStart, iComputeTileSize, jComputeTileSize, kComputeTileSize,
            vectorLen, fullUnrollAndJam);
        }, /* else has partial tiles */ [&](const AffineBuilderKrnlMem &createAffine) {
          genScalar(createAffine, matmulOp, TmpScalarProd, elementType, aStart, bStart, cStart,
            iTrip, jTrip, kTrip, /*unroll*/ false);
        });
        // clang-format on
      } else {
        Value TmpSimdC = allocForGenSimdMatMat(create.affineKMem, elementType,
            iComputeTileSize, jComputeTileSize, kComputeTileSize, vectorLen,
            fullUnrollAndJam);
        Value TmpScalarC = allocForGenScalar(create.affineKMem, elementType,
            iTrip, jPartialTrip, kTrip, /*unroll*/ false);
        // clang-format off
        create.affineKMem.ifThenElseIE(indexScope, allFullTiles,
          /* then full tiles */ [&](const AffineBuilderKrnlMem &createAffine) {
          genSimdMatMat(createAffine, matmulOp, TmpSimdC, elementType, aStart, bStart,
             cStart, iComputeTileSize, jComputeTileSize, kComputeTileSize,
            vectorLen, fullUnrollAndJam);
          }, 
          /* Else has some partial tiles */ 
          [&](const AffineBuilderKrnlMem &createAffine) {
          // Trip regardless of full/partial for N & K
          // Test if SIMD dim (M) is full.
          createAffine.ifThenElseIE(indexScope, jFullTiles,
            /* full SIMD */ [&](const AffineBuilderKrnlMem &createAffine) {
            genSimdMatMat(createAffine, matmulOp, TmpSimdC, elementType, aStart, bStart,
               cStart, iTrip, jComputeTileSize, kTrip, vectorLen, /*unroll*/ false);
          }, /* else partial SIMD */ [&](const AffineBuilderKrnlMem &createAffine) {
            genScalar(createAffine, matmulOp, TmpScalarC, elementType, aStart, bStart, cStart,
              iTrip, jPartialTrip, kTrip, /*unroll*/ false);
          });
        });
        // clang-format on
      }
    } else {
      // Scalar code generator.
      Value TmpThenC =
          allocForGenScalar(create.affineKMem, elementType, iComputeTileSize,
              jComputeTileSize, kComputeTileSize, fullUnrollAndJam);
      Value TmpElseC = allocForGenScalar(
          create.affineKMem, elementType, iTrip, jTrip, kTrip, false);
      // clang-format off
      create.affineKMem.ifThenElseIE(indexScope, allFullTiles,
        /* then full */ [&](const AffineBuilderKrnlMem &createAffine) {
        genScalar(createAffine, matmulOp, TmpThenC, elementType, aStart, bStart, cStart,
          iComputeTileSize, jComputeTileSize, kComputeTileSize,
          fullUnrollAndJam);
      }, /* else partial */ [&](const AffineBuilderKrnlMem &createAffine) {
        genScalar(createAffine, matmulOp, TmpElseC, elementType, aStart, bStart, cStart,
          iTrip, jTrip, kTrip, false);
      });
      // clang-format on
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  Value allocForGenScalar(const AffineBuilderKrnlMem &createAffine,
      Type elementType, IndexExpr I, IndexExpr J, IndexExpr K,
      bool unrollJam) const {
    // Get operands.
    MemRefBuilder createMemRef(createAffine);
    int64_t unrollFactor = (unrollJam && J.isLiteral()) ? J.getLiteral() : 1;
    // Have to privatize CTmpType by unroll factor (1 if none).
    MemRefType CTmpType = MemRefType::get({unrollFactor}, elementType);
    assert(BUFFER_ALIGN >= gDefaultAllocAlign);

    if (parallelEnabled)
      return createMemRef.alignedAlloc(CTmpType, BUFFER_ALIGN);
    return createMemRef.alignedAlloca(CTmpType, BUFFER_ALIGN);
  }

  void genScalar(const AffineBuilderKrnlMem &createAffine, KrnlMatMulOp op,
      Value TmpC, Type elementType, ArrayRef<IndexExpr> aStart,
      ArrayRef<IndexExpr> bStart, ArrayRef<IndexExpr> cStart, IndexExpr I,
      IndexExpr J, IndexExpr K, bool unrollJam) const {
    // Get operands.
    KrnlMatMulOpAdaptor operandAdaptor(op);
    MemRefBuilder createMemRef(createAffine);

    Value A(operandAdaptor.getA()), B(operandAdaptor.getB()),
        C(operandAdaptor.getC());
    int64_t unrollFactor = (unrollJam && J.isLiteral()) ? J.getLiteral() : 1;

    // For i, j loops.
    LiteralIndexExpr zeroIE(0);
    Value jSaved;
    createAffine.forLoopIE(zeroIE, I, 1,
        [&](const AffineBuilderKrnlMem &createAffine, ValueRange loopInd) {
          Value i = loopInd[0];
          createAffine.forLoopIE(zeroIE, J, 1,
              [&](const AffineBuilderKrnlMem &createAffine,
                  ValueRange loopInd) {
                MathBuilder createMath(createAffine);
                Value j = loopInd[0];
                // Defines induction variables, and possibly initialize C.
                jSaved = j;
                // Alloc and init temp c storage.
                Value initVal = createAffine.loadIE(C, cStart, {i, j});
                Value tmpCAccess = (unrollFactor > 1) ? j : zeroIE.getValue();
                // TTmpC() = affine_load(C, cAccess);
                createAffine.store(initVal, TmpC, tmpCAccess);
                // Sum over k.
                createAffine.forLoopIE(zeroIE, K, 1,
                    [&](const AffineBuilderKrnlMem &createAffine,
                        ValueRange loopInd) {
                      MathBuilder createMath(createAffine);
                      Value k = loopInd[0];
                      Value a = createAffine.loadIE(A, aStart, {i, k});
                      Value b = createAffine.loadIE(B, bStart, {k, j});
                      Value res = createMath.mul(a, b);
                      res = createMath.add(
                          res, createAffine.load(TmpC, tmpCAccess));
                      // TTmpC() = a * b + TTmpC();
                      createAffine.store(res, TmpC, tmpCAccess);
                    });
                // Store temp result into C(i, j)
                Value finalVal = createAffine.load(TmpC, tmpCAccess);
                createAffine.storeIE(finalVal, C, cStart, {i, j});
              });
        });
    if (unrollJam && J.isLiteral()) {
      UnrollAndJamRecord record(
          affine::getForInductionVarOwner(jSaved), J.getLiteral());
      getUnrollAndJamList(op)->emplace_back(record);
    }
  }

  Value allocForGenSimdMatVect(const AffineBuilderKrnlMem &createAffine,
      Type elementType, IndexExpr I, IndexExpr J, IndexExpr K,
      IndexExpr vectorLen, bool unrollJam) const {
    // can simdize only if I & K is compile time
    assert(I.isLiteral() && K.isLiteral() && vectorLen.isLiteral() &&
           "can only simdize with compile time "
           "blocking factor on simd axis");
    MultiDialectBuilder<VectorBuilder, MemRefBuilder> create(createAffine);
    int64_t iLit(I.getLiteral()), VL(vectorLen.getLiteral());
    int64_t archVL = create.vec.getArchVectorLength(elementType);

    // Generate the vector type conversions.
    assert(VL == archVL && "vector length and VL must be identical for now");
    VectorType vecType = VectorType::get({VL}, elementType);
    int64_t iUnrollFactor = iLit;
    assert(iUnrollFactor % VL == 0 && "i blocking should be a multiple of VL");

    // Have to privatize CTmpType by unroll factor.
    MemRefType CTmpType = MemRefType::get({iUnrollFactor}, vecType);
    assert(BUFFER_ALIGN >= gDefaultAllocAlign &&
           "alignment of buffers cannot be smaller than the default alignment "
           "(which is set for SIMD correctness");
    // Ok to use an alloca here because hoisting will take it out of the loop,
    // as it is now generated before the scf.if which precluded the migration to
    // outside the loops.

    // But at this time, if parallel is enabled, alloca would be stuck inside of
    // the parallel loop, which is not great. TODO: migrate alloca from inside
    // the parallel loop to the OMP parallel region before the loop.
    // Grep for this pattern in all 3 instances of "parallelEnabled".

    if (parallelEnabled)
      return create.mem.alignedAlloc(CTmpType, BUFFER_ALIGN);
    return create.mem.alignedAlloca(CTmpType, BUFFER_ALIGN);
  }

  // Initially, simdize with full K vector length.
  void genSimdMatVect(const AffineBuilderKrnlMem &createAffine, KrnlMatMulOp op,
      Value TmpProd, Type elementType, ArrayRef<IndexExpr> aStart,
      ArrayRef<IndexExpr> bStart, ArrayRef<IndexExpr> cStart, IndexExpr I,
      IndexExpr J, IndexExpr K, IndexExpr vectorLen, bool unrollJam) const {
    // can simdize only if I & K is compile time
    assert(I.isLiteral() && K.isLiteral() && vectorLen.isLiteral() &&
           "can only simdize with compile time "
           "blocking factor on simd axis");
    MultiDialectBuilder<MathBuilder, VectorBuilder, AffineBuilderKrnlMem,
        MemRefBuilder, KrnlBuilder>
        create(createAffine);
    int64_t iLit(I.getLiteral()), VL(vectorLen.getLiteral());
    int64_t archVL = create.vec.getArchVectorLength(elementType);
    // Get operands.
    KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(op);
    Value A(operandAdaptor.getA()), B(operandAdaptor.getB()),
        C(operandAdaptor.getC());

    // Generate the vector type conversions.
    assert(VL == archVL && "vector length and VL must be identical for now");
    VectorType vecType = VectorType::get({VL}, elementType);
    int64_t iUnrollFactor = iLit;
    assert(iUnrollFactor % VL == 0 && "i blocking should be a multiple of VL");

    // Init with zero.
    Value fZero = create.math.constant(elementType, 0);
    Value vFZero = create.vec.broadcast(vecType, fZero);
    create.krnl.memset(TmpProd, vFZero);
    LiteralIndexExpr zeroIE(0);
    Value iZero = create.math.constantIndex(0);

    create.affineKMem.forLoopIE(zeroIE, K, VL,
        [&](const AffineBuilderKrnlMem &createAffine, ValueRange loopInd) {
          MultiDialectBuilder<MathBuilder, VectorBuilder> create(createAffine);
          Value k = loopInd[0];
          // Iterates over the I indices (K is SIMD dim).
          // First compute A[i,k]*B[k, 1] for i=0..iUnrollFactor explicitly.
          // We reuse B[k][0] vector for each iteration of i.
          Value vb = create.vec.loadIE(vecType, B, bStart, {k, iZero});
          // Generate computation for each i, manually unrolled for simplicity.
          for (int64_t i = 0; i < iUnrollFactor; ++i) {
            Value iVal = create.math.constantIndex(i);
            Value va = create.vec.loadIE(vecType, A, aStart, {iVal, k});
            Value vTmpProd = create.vec.load(vecType, TmpProd, {iVal});
            Value vres;
            if (isa<FloatType>(elementType)) {
              vres = create.vec.fma(va, vb, vTmpProd);
            } else {
              vres = create.math.mul(va, vb);
              vres = create.math.add(vres, vTmpProd);
            }
            create.vec.store(vres, TmpProd, {iVal});
          }
        });

    // Reduce each SIMD vector of length VL using a SIMD parallel reduction.
    SmallVector<Value, 8> vProdList, vReductionList;
    for (int64_t i = 0; i < iUnrollFactor; ++i) {
      Value iVal = create.math.constantIndex(i);
      Value vTmpProd = create.vec.load(vecType, TmpProd, {iVal});
      vProdList.emplace_back(vTmpProd);
    }
    create.vec.multiReduction(
        vProdList,
        [create](Value a, Value b) -> Value { return create.math.add(a, b); },
        vReductionList);
    // For each reduction in the list (vector of VL length), load C, add
    // reduction, and store C.
    uint64_t size = vReductionList.size();
    for (uint64_t i = 0; i < size; ++i) {
      // IndexExpr::getValues(cStart, cAccess);
      Value iVal = create.math.constantIndex(i * VL);
      Value vc = create.vec.loadIE(vecType, C, cStart, {iVal, iZero});
      vc = create.math.add(vc, vReductionList[i]);
      create.vec.storeIE(vc, C, cStart, {iVal, iZero});
    }
  }

  Value allocForGenSimdMatMat(const AffineBuilderKrnlMem &createAffine,
      Type elementType, IndexExpr I, IndexExpr J, IndexExpr K,
      IndexExpr vectorLen, bool unrollJam) const {
    // can simdize only if K is compile time
    MultiDialectBuilder<MemRefBuilder> create(createAffine);

    // Generate the vector type conversions.
    int64_t VL = vectorLen.getLiteral();
    VectorType vecType = VectorType::get({VL}, elementType);
    int64_t unrollFactor = (unrollJam && I.isLiteral()) ? I.getLiteral() : 1;
    // Have to privatize CTmpType by unroll factor (1 if none).
    MemRefType CTmpType = MemRefType::get({unrollFactor}, vecType);
    assert(BUFFER_ALIGN >= gDefaultAllocAlign);
    // Ok to use an alloca here because hoisting will take it out of the loop,
    // as it is now generated before the scf.if which precluded the migration to
    // outside the loops.

    // But at this time, if parallel is enabled, alloca would be stuck inside of
    // the parallel loop, which is not great. TODO: migrate alloca from inside
    // the parallel loop to the OMP parallel region before the loop.

    if (parallelEnabled)
      return create.mem.alignedAlloc(CTmpType, BUFFER_ALIGN);
    return create.mem.alignedAlloca(CTmpType, BUFFER_ALIGN);
  }

  // Simdize along J / memory rows in B and C.
  void genSimdMatMat(const AffineBuilderKrnlMem &createAffine, KrnlMatMulOp op,
      Value TmpC, Type elementType, ArrayRef<IndexExpr> aStart,
      ArrayRef<IndexExpr> bStart, ArrayRef<IndexExpr> cStart, IndexExpr I,
      IndexExpr J, IndexExpr K, IndexExpr vectorLen, bool unrollJam) const {
    // can simdize only if K is compile time
    assert(J.isLiteral() &&
           "can only simdize with compile time blocking factor on simd axis");
    MultiDialectBuilder<MathBuilder, MemRefBuilder> create(createAffine);

    // Get operands.
    KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(op);
    Value A(operandAdaptor.getA()), B(operandAdaptor.getB()),
        C(operandAdaptor.getC());

    // Generate the vector type conversions.
    int64_t VL = vectorLen.getLiteral();
    VectorType vecType = VectorType::get({VL}, elementType);
    int64_t unrollFactor = (unrollJam && I.isLiteral()) ? I.getLiteral() : 1;

    // Iterates over the I indices (j are simd dim).
    Value iSaved, kSaved;
    LiteralIndexExpr zeroIE(0);
    Value iZero = create.math.constantIndex(0);

    createAffine.forLoopIE(zeroIE, I, 1,
        [&](const AffineBuilderKrnlMem &createAffine, ValueRange loopInd) {
          MultiDialectBuilder<MathBuilder, VectorBuilder> create(createAffine);
          Value i = loopInd[0];
          iSaved = i; // Saved for unroll and jam.
          // Alloc temp vector TmpC and save C(i)/0.0 into it.
          Value initVal = create.vec.loadIE(vecType, C, cStart, {i, iZero});
          Value tmpCAccess = (unrollFactor > 1) ? i : zeroIE.getValue();
          createAffine.store(initVal, TmpC, tmpCAccess);
          // Sum over k.
          createAffine.forLoopIE(zeroIE, K, 1,
              [&](const AffineBuilderKrnlMem &createAffine,
                  ValueRange loopInd) {
                MultiDialectBuilder<MathBuilder, VectorBuilder> create(
                    createAffine);
                Value k = loopInd[0];
                kSaved = k;
                Value a = createAffine.loadIE(A, aStart, {i, k});
                Value va = create.vec.broadcast(vecType, a);
                Value vb = create.vec.loadIE(vecType, B, bStart, {k, iZero});
                // TTmpC() = vector_fma(va, vb, TTmpC());
                Value tmpVal = createAffine.load(TmpC, tmpCAccess);
                Value res;
                if (isa<FloatType>(elementType)) {
                  res = create.vec.fma(va, vb, tmpVal);
                } else {
                  res = create.math.mul(va, vb);
                  res = create.math.add(res, tmpVal);
                }
                createAffine.store(res, TmpC, tmpCAccess);
              });
          // Store temp result into C(i)
          Value tmpResults = createAffine.load(TmpC, tmpCAccess);
          int64_t JLit = J.getLiteral();
          if (JLit != VL) {
            // create vector constant
            SmallVector<int64_t, 8> mask;
            for (int64_t i = 0; i < VL; i++)
              mask.emplace_back((i < JLit) ? i : VL + i);
            // permute
            Value originalCvec =
                create.vec.loadIE(vecType, C, cStart, {i, iZero});
            tmpResults = create.vec.shuffle(tmpResults, originalCvec, mask);
          }
          create.vec.storeIE(tmpResults, C, cStart, {i, iZero});
        });

    if (unrollJam && (I.isLiteral() || K.isLiteral())) {
      auto list = getUnrollAndJamList(op);
      if (K.isLiteral()) {
        int64_t kUnroll = K.getLiteral();
        // We know there is no unrolling along I, make a bigger cutoff.
        int64_t cutoff = (!I.isLiteral() || I.getLiteral() < 2) ? 8 : 4;
        if (kUnroll >= cutoff) {
          // When kUnroll is too big, reduce it by a divisor.
          for (int64_t m = cutoff; m >= 1; --m) {
            if (kUnroll % m == 0) {
              kUnroll = m;
              break;
            }
          }
        }
        if (kUnroll > 1) {
          LLVM_DEBUG(
              llvm::dbgs() << "Matmul: unroll k by " << kUnroll << "\n";);
          UnrollAndJamRecord record(
              affine::getForInductionVarOwner(kSaved), kUnroll);
          list->emplace_back(record);
        }
      }
      if (I.isLiteral() && I.getLiteral() > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Matmul: unroll i by " << (int)I.getLiteral() << "\n");
        UnrollAndJamRecord record(
            affine::getForInductionVarOwner(iSaved), I.getLiteral());
        list->emplace_back(record);
      }
    }
  }

  UnrollAndJamList *getUnrollAndJamList(Operation *op) const {
    Operation *currFuncOp = getContainingFunction(op);
    assert(currFuncOp && "function expected");
    const std::lock_guard<std::mutex> lock(unrollAndJamMutex);
    UnrollAndJamList *currUnrollAndJamList = unrollAndJamMap[currFuncOp];
    assert(currUnrollAndJamList && "expected list for function");
    return currUnrollAndJamList;
  }
}; // namespace krnl

void populateLoweringKrnlMatmultOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx, bool parallelEnabled) {
  patterns.insert<KrnlMatmulLowering>(typeConverter, ctx, parallelEnabled);
}

} // namespace krnl
} // namespace onnx_mlir
