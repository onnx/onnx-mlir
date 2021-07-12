/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Matmul.cpp - Lowering Matmul Op --------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/MLIRDialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXMatMulOpLowering : public ConversionPattern {
  ONNXMatMulOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXMatMulOp::getOperationName(), 1, ctx) {}

  // Handle the generic cases, including when there are broadcasts.
  void replaceGenericMatmul(ONNXMatMulOp &matMulOp,
      ONNXMatMulOpAdaptor &operandAdaptor, Type elementType,
      ONNXMatMulOpShapeHelper &shapeHelper, Value alloc, Value fzero,
      ConversionPatternRewriter &rewriter, Location loc) const {

    // Non-reduction loop iterations: output-rank.
    KrnlBuilder createKrnl(rewriter, loc);
    int outerloopNum = shapeHelper.dimsForOutput(0).size();
    ValueRange outerLoops = createKrnl.defineLoops(outerloopNum);
    SmallVector<IndexExpr, 4> outerLbs(outerloopNum, LiteralIndexExpr(0));
    createKrnl.iterateIE(outerLoops, outerLoops, outerLbs,
        shapeHelper.dimsForOutput(0), {},
        [&](KrnlBuilder &createKrnl, ValueRange args) {
          ValueRange outerIndices = createKrnl.getInductionVarValue(outerLoops);
          ImplicitLocOpBuilder lb(createKrnl.getLoc(), createKrnl.getBuilder());
          Value reductionVal =
              lb.create<memref::AllocaOp>(MemRefType::get({}, elementType));
          createKrnl.store(fzero, reductionVal);
          int aRank = shapeHelper.aDims.size();
          int bRank = aRank; // Add for better readability.
          ValueRange innerLoop = lb.create<KrnlDefineLoopsOp>(1).getResults();
          Value innerUb = shapeHelper.aDims[aRank - 1].getValue();
          Value izero = lb.create<ConstantIndexOp>(0);
          createKrnl.iterate(innerLoop, innerLoop, {izero}, {innerUb}, {},
              [&](KrnlBuilder &createKrnl, ValueRange args) {
                ValueRange innerIndex =
                    createKrnl.getInductionVarValue(innerLoop);
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
                    } else if (i == outerloopNum) {
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
                Value loadedA = createKrnl.load(operandAdaptor.A(), aAccessFct);
                Value loadedB = createKrnl.load(operandAdaptor.B(), bAccessFct);
                Value loadedY = createKrnl.load(reductionVal);
                MathBuilder createMath(createKrnl);
                Value AB = createMath.mul(loadedA, loadedB);
                Value accumulated = createMath.add(loadedY, AB);
                createKrnl.store(accumulated, reductionVal);
              });
          Value accumulated = createKrnl.load(reductionVal);
          createKrnl.store(accumulated, alloc, outerIndices);
        });
  }

  // Handle the cases with 2x2 matrices both for A, B, and C without broadcast.
  // Implementation here uses the efficient 1d tiling plus kernel substitution.
  void replace2x2Matmul2d(ONNXMatMulOp &matMulOp,
      ONNXMatMulOpAdaptor &operandAdaptor, Type elementType,
      ONNXMatMulOpShapeHelper &shapeHelper, Value alloc, Value zeroVal,
      ConversionPatternRewriter &rewriter, Location loc) const {

    // Prepare: loop bounds and zero
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(alloc);
    KrnlBuilder createKrnl(rewriter, loc);
    ImplicitLocOpBuilder lb(loc, rewriter);
    Value zero = lb.create<ConstantIndexOp>(0);
    Value one = lb.create<ConstantIndexOp>(1);
    Value I = lb.createOrFold<memref::DimOp>(C, zero);
    Value J = lb.createOrFold<memref::DimOp>(C, one);
    Value K = lb.createOrFold<memref::DimOp>(A, one);

    // Initialize alloc/C to zero.
    ValueRange zLoop = createKrnl.defineLoops(2);
    createKrnl.iterate(zLoop, zLoop, {zero, zero}, {I, J}, {},
        [&](KrnlBuilder &createKrnl, ValueRange args) {
          ValueRange indices = createKrnl.getInductionVarValue(zLoop);
          createKrnl.store(zeroVal, alloc, indices);
        });

    // Compute.
    // Define blocking, with simdization along the j axis.
    const int64_t iRegTile(4), jRegTile(8), kRegTile(4);
    // I, J, K loop.
    ValueRange origLoop = createKrnl.defineLoops(3);
    Value ii(origLoop[0]), jj(origLoop[1]), kk(origLoop[2]);
    // Define blocked loop and permute.
    ValueRange iRegBlock = createKrnl.block(ii, iRegTile);
    Value ii1(iRegBlock[0]), ii2(iRegBlock[1]);
    ValueRange jRegBlock = createKrnl.block(jj, jRegTile);
    Value jj1(jRegBlock[0]), jj2(jRegBlock[1]);
    ValueRange kRegBlock = createKrnl.block(kk, kRegTile);
    Value kk1(kRegBlock[0]), kk2(kRegBlock[1]);
    createKrnl.permute({ii1, ii2, jj1, jj2, kk1, kk2}, {0, 3, 1, 4, 2, 5});
    createKrnl.iterate({ii, jj, kk}, {ii1, jj1, kk1}, {zero, zero, zero},
        {I, J, K}, {}, [&](KrnlBuilder &createKrnl, ValueRange args) {
          ValueRange indices = createKrnl.getInductionVarValue({ii1, jj1, kk1});
          Value i1(indices[0]), j1(indices[1]), k1(indices[2]);
          createKrnl.matmul(A, {zero, zero}, B, {zero, zero}, C, {zero, zero},
              {ii2, jj2, kk2}, {i1, j1, k1}, {I, J, K},
              {iRegTile, jRegTile, kRegTile}, {}, {}, {}, true, true, false);
        });
  }

  // Handle the cases with 2x2 matrices both for A, B, and C without broadcast.
  // Implementation here uses the efficient 2d tiling plus kernel substitution.

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Get shape.
    ONNXMatMulOpAdaptor operandAdaptor(operands);
    ONNXMatMulOp matMulOp = llvm::cast<ONNXMatMulOp>(op);
    Location loc = ONNXLoc<ONNXMatMulOp>(op);
    ONNXMatMulOpShapeHelper shapeHelper(&matMulOp, rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    LogicalResult shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = outputMemRefType.getElementType();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Get the constants: zero.
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);

    Value A(operandAdaptor.A()), B(operandAdaptor.B());
    auto aRank = A.getType().cast<MemRefType>().getShape().size();
    auto bRank = B.getType().cast<MemRefType>().getShape().size();
    if (aRank == 2 && bRank == 2) {
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

void populateLoweringONNXMatMulOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLowering>(ctx);
}
