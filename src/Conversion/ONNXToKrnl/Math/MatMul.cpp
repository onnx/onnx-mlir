/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Matmul.cpp - Lowering Matmul Op --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

#define DEBUG_TYPE "matmul"
static constexpr int32_t DISABLE_MAT_VEC_PRODUCT = 0;

using namespace mlir;

namespace onnx_mlir {

struct ONNXMatMulOpLowering : public ConversionPattern {
  ONNXMatMulOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableTiling)
      : ConversionPattern(
            typeConverter, mlir::ONNXMatMulOp::getOperationName(), 1, ctx),
        enableTiling(enableTiling) {}
  bool enableTiling;
  // Handle the generic cases, including when there are broadcasts.
  void replaceGenericMatmul(ONNXMatMulOp &matMulOp,
      ONNXMatMulOpAdaptor &operandAdaptor, Type elementType,
      ONNXMatMulOpShapeHelper &shapeHelper, Value alloc, Value fZero,
      ConversionPatternRewriter &rewriter, Location loc) const {

    // Define loops and bounds.
    KrnlBuilder createKrnl(rewriter, loc);
    int outerLoopNum = shapeHelper.dimsForOutput().size();
    int totLoopNum = outerLoopNum + 1; // Add reduction inner loop.
    ValueRange loopDef = createKrnl.defineLoops(totLoopNum);
    SmallVector<IndexExpr, 4> loopLbs(totLoopNum, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> loopUbs; // All dimsForOutputs, plus reduction.
    SmallVector<Value, 4> outerLoops;  // All but the last loop def.
    for (int i = 0; i < outerLoopNum; ++i) {
      loopUbs.emplace_back(shapeHelper.dimsForOutput()[i]);
      outerLoops.emplace_back(loopDef[i]);
    }
    int aRank = shapeHelper.aDims.size();
    int bRank = aRank; // Add for better readability.
    IndexExpr innerUb = shapeHelper.aDims[aRank - 1];
    loopUbs.emplace_back(innerUb);
    SmallVector<Value, 1> innerLoop{loopDef[totLoopNum - 1]}; // Last loop def.

    // Non-reduction loop iterations: output-rank.
    createKrnl.iterateIE(loopDef, outerLoops, loopLbs, loopUbs,
        [&](KrnlBuilder &createKrnl, ValueRange outerIndices) {
          MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
              createKrnl);
          // Single scalar, no need for default alignment.
          Value reductionVal =
              create.mem.alignedAlloca(MemRefType::get({}, elementType));
          create.krnl.store(fZero, reductionVal);
          // Inner loop for reduction.
          create.krnl.iterate({}, innerLoop, {}, {},
              [&](KrnlBuilder &createKrnl, ValueRange innerIndex) {
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
                    create.krnl.load(operandAdaptor.A(), aAccessFct);
                Value loadedB =
                    create.krnl.load(operandAdaptor.B(), bAccessFct);
                Value loadedY = create.krnl.load(reductionVal);
                Value AB = create.math.mul(loadedA, loadedB);
                Value accumulated = create.math.add(loadedY, AB);
                create.krnl.store(accumulated, reductionVal);
              });
          Value accumulated = create.krnl.load(reductionVal);
          create.krnl.store(accumulated, alloc, outerIndices);
        });
  }

  void computeTileSizeForMatMatProduct(DimIndexExpr dimI, DimIndexExpr dimJ,
      DimIndexExpr dimK, int64_t &iRegTile, int64_t &jRegTile,
      int64_t &kRegTile, bool &simdize) const {

    // Default values
    iRegTile = 4;
    jRegTile = 8;
    kRegTile = 8; // SIMD dim.

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
    LLVM_DEBUG({
      llvm::dbgs() << "MatMul mat: Tiling I " << iRegTile << ", J " << jRegTile
                   << ", K " << kRegTile << ", simd " << simdize << "\n";
    });
  }

  void computeTileSizeForMatVectProduct(int64_t mVL, DimIndexExpr dimI,
      DimIndexExpr dimJ, DimIndexExpr dimK, int64_t &iRegTile,
      int64_t &jRegTile, int64_t &kRegTile, bool &simdize) const {

    // Default values.
    // Right can only tile i and k by (possibly distinct) multiple of mVL.
    iRegTile = 2 * mVL; // SIMD dim during multi-reduction.
    jRegTile = 1;
    kRegTile = 16 * mVL; // SIMD dim during multiplication.

    if (dimK.isLiteral()) {
      int64_t constK = dimK.getLiteral();
      // Register tile in the I Dim is really for the reduction. The
      // computations will be further tiled to a multiple of mVL inside
      // krnl.matmul.
      kRegTile = (constK / mVL) * mVL; // largest multiple
      if (kRegTile > 64 * mVL) {
        kRegTile = 64 * mVL;
        LLVM_DEBUG({ llvm::dbgs() << "MatMul Vec: cap tiling k\n"; });
      } else if (kRegTile < mVL) {
        // Not enough data, can only support i/k reg tile of 4.
        LLVM_DEBUG({ llvm::dbgs() << "MatMul Vec: disable k\n"; });
        simdize = false;
        kRegTile = 1;
      }
    }
    if (dimI.isLiteral()) {
      int64_t constI = dimI.getLiteral();
      if (constI < iRegTile) {
        iRegTile = (constI / mVL) * mVL; // largest multiple
        if (iRegTile < mVL) {
          // Not enough data, can only support i/k reg tile of 4.
          LLVM_DEBUG({ llvm::dbgs() << "MatMul Vec: disable i\n"; });
          simdize = false;
          iRegTile = 1;
        }
      }
    }
    LLVM_DEBUG({
      llvm::dbgs() << "MatMul vec: Tiling I " << iRegTile << ", J " << jRegTile
                   << ", K " << kRegTile << ", simd " << simdize << "\n";
    });
  }

  // Handle the cases with 2x2 matrices both for A, B, and C without
  // broadcast. Implementation here uses the efficient 1d tiling plus kernel
  // substitution.
  void replace2x2Matmul2d(ONNXMatMulOp &matMulOp,
      ONNXMatMulOpAdaptor &operandAdaptor, Type elementType,
      ONNXMatMulOpShapeHelper &shapeHelper, Value alloc, Value zeroVal,
      ConversionPatternRewriter &rewriter, Location loc) const {
    // Prepare: loop bounds and zero
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(alloc);
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder, VectorBuilder>
        create(rewriter, loc);
    Value zero = create.math.constantIndex(0);
    Value I = create.mem.dim(C, 0);
    Value J = create.mem.dim(C, 1);
    Value K = create.mem.dim(A, 1);

    // Initialize alloc/C to zero.
    create.krnl.memset(alloc, zeroVal);
    bool simdize = true;

    // Define blocking, with simdization along the j axis.
    DimIndexExpr dimI(I), dimJ(J), dimK(K);
    int64_t iRegTile, jRegTile, kRegTile;
    bool isMatVectorProduct =
        !DISABLE_MAT_VEC_PRODUCT && dimJ.isLiteral() && dimJ.getLiteral() == 1;
    if (isMatVectorProduct) {
      int64_t mVL = create.vec.getMachineVectorLength(elementType);
      computeTileSizeForMatVectProduct(
          mVL, dimI, dimJ, dimK, iRegTile, jRegTile, kRegTile, simdize);
    } else {
      computeTileSizeForMatMatProduct(
          dimI, dimJ, dimK, iRegTile, jRegTile, kRegTile, simdize);
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
    create.krnl.iterate({ii, jj, kk}, {ii1, jj1, kk1}, {zero, zero, zero},
        {I, J, K}, [&](KrnlBuilder &createKrnl, ValueRange indices) {
          Value i1(indices[0]), j1(indices[1]), k1(indices[2]);
          createKrnl.matmul(A, {zero, zero}, B, {zero, zero}, C, {zero, zero},
              {ii2, jj2, kk2}, {i1, j1, k1}, {I, J, K},
              {iRegTile, jRegTile, kRegTile}, {}, {}, {}, simdize,
              /*unroll*/ true, /*overcompute*/ false);
        });
  }

  // Handle the cases with 2x2 matrices both for A, B, and C without
  // broadcast. Implementation here uses the efficient 2d tiling plus kernel
  // substitution.

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get shape.
    ONNXMatMulOpAdaptor operandAdaptor(operands);
    ONNXMatMulOp matMulOp = llvm::cast<ONNXMatMulOp>(op);
    Location loc = ONNXLoc<ONNXMatMulOp>(op);
    ONNXMatMulOpShapeHelper shapeHelper(&matMulOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = outputMemRefType.getElementType();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput());

    // Get the constants: zero.
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);

    Value A(operandAdaptor.A()), B(operandAdaptor.B());
    auto aRank = A.getType().cast<MemRefType>().getShape().size();
    auto bRank = B.getType().cast<MemRefType>().getShape().size();
    if (enableTiling && aRank == 2 && bRank == 2) {
      // Optimized Matmul only when 2D and allowed to tile and unroll.
      replace2x2Matmul2d(matMulOp, operandAdaptor, elementType, shapeHelper,
          alloc, zero, rewriter, loc);
    } else {
      replaceGenericMatmul(matMulOp, operandAdaptor, elementType, shapeHelper,
          alloc, zero, rewriter, loc);
    }
    // Done.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXMatMulOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableTiling) {
  patterns.insert<ONNXMatMulOpLowering>(typeConverter, ctx, enableTiling);
}

} // namespace onnx_mlir
