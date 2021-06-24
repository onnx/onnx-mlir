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
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

// TODO rename to MLIR file
#include "src/Dialect/ONNX/TmpMlirUtils.hpp"

using namespace mlir;

struct ONNXMatMulOpLowering : public ConversionPattern {
  ONNXMatMulOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXMatMulOp::getOperationName(), 1, ctx) {}

  // Handle the generic cases, including when there are broadcasts.
  void replaceGenericMatmul(ONNXMatMulOp &matMulOp,
      ONNXMatMulOpAdaptor &operandAdaptor, Type elementType,
      ONNXMatMulOpShapeHelper &shapeHelper, Value alloc, Value fzero,
      ConversionPatternRewriter &rewriter, Location loc) const {
    ImplicitLocOpBuilder lb(loc, rewriter);

    // Non-reduction loop iterations: output-rank.
    int outerloopNum = shapeHelper.dimsForOutput(0).size();
    ValueRange outerLoops =
        lb.create<KrnlDefineLoopsOp>(outerloopNum).getResults();
    SmallVector<IndexExpr, 4> outerLbs(outerloopNum, LiteralIndexExpr(0));
    lb.create<KrnlIterateOp>(outerLoops, outerLoops, outerLbs,
        shapeHelper.dimsForOutput(0), ValueRange{},
        [&](ImplicitLocOpBuilder &lb, ValueRange args) {
          ValueRange outerIndices =
              lb.create<KrnlGetInductionVariableValueOp>(outerLoops)
                  .getResults();
          Value reductionVal =
              lb.create<memref::AllocaOp>(MemRefType::get({}, elementType));
          lb.create<KrnlStoreOp>(fzero, reductionVal);
          int aRank = shapeHelper.aDims.size();
          int bRank = aRank; // Add for better readability.
          ValueRange innerLoop = lb.create<KrnlDefineLoopsOp>(1).getResults();
          ValueRange ub{shapeHelper.aDims[aRank - 1].getValue()};
          Value izero = lb.create<ConstantIndexOp>(0);
          lb.create<KrnlIterateOp>(innerLoop, innerLoop, ValueRange{izero}, ub,
              ValueRange{}, [&](ImplicitLocOpBuilder &lb, ValueRange args) {
                ValueRange innerIndex =
                    lb.create<KrnlGetInductionVariableValueOp>(innerLoop)
                        .getResults();
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
                Value loadedA =
                    lb.create<KrnlLoadOp>(operandAdaptor.A(), aAccessFct);
                Value loadedB =
                    lb.create<KrnlLoadOp>(operandAdaptor.B(), bAccessFct);
                Value loadedY = lb.create<KrnlLoadOp>(reductionVal);
                Value AB = lb.create<MulFOp>(loadedA, loadedB);
                Value accumulated = lb.create<AddFOp>(loadedY, AB);
                lb.create<KrnlStoreOp>(accumulated, reductionVal);
              });
          Value accumulated = lb.create<KrnlLoadOp>(reductionVal);
          lb.create<KrnlStoreOp>(accumulated, alloc, outerIndices);
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
    ImplicitLocOpBuilder lb(loc, rewriter);
    Value zero = lb.create<ConstantIndexOp>(0);
    Value one = lb.create<ConstantIndexOp>(1);
    Value I = lb.createOrFold<memref::DimOp>(C, zero);
    Value J = lb.createOrFold<memref::DimOp>(C, one);
    Value K = lb.createOrFold<memref::DimOp>(A, one);

    // Initialize alloc/C to zero.
    ValueRange zLoop = lb.create<KrnlDefineLoopsOp>(2).getResults();
    lb.create<KrnlIterateOp>(zLoop, zLoop, ValueRange{zero, zero},
        ValueRange{I, J}, ValueRange{},
        [&](ImplicitLocOpBuilder &lb, ValueRange args) {
          ValueRange indices =
              lb.create<KrnlGetInductionVariableValueOp>(zLoop).getResults();
          lb.create<KrnlStoreOp>(zeroVal, alloc, indices);
        });

    // Compute.
    // Define blocking, with simdization along the j axis.
    const int64_t iRegTile(4), jRegTile(8), kRegTile(4);
    // I, J, K loop.
    ValueRange origLoop = lb.create<KrnlDefineLoopsOp>(3).getResults();
    Value ii(origLoop[0]), jj(origLoop[1]), kk(origLoop[2]);
    // Define blocked loop and permute.
    ValueRange iRegBlock = lb.create<KrnlBlockOp>(ii, iRegTile).getResults();
    Value ii1(iRegBlock[0]), ii2(iRegBlock[1]);
    ValueRange jRegBlock = lb.create<KrnlBlockOp>(jj, jRegTile).getResults();
    Value jj1(jRegBlock[0]), jj2(jRegBlock[1]);
    ValueRange kRegBlock = lb.create<KrnlBlockOp>(kk, kRegTile).getResults();
    Value kk1(kRegBlock[0]), kk2(kRegBlock[1]);
    lb.create<KrnlPermuteOp>(ValueRange{ii1, ii2, jj1, jj2, kk1, kk2},
        ArrayRef<int64_t>{0, 3, 1, 4, 2, 5});

    lb.create<KrnlIterateOp>(ValueRange({ii, jj, kk}),
        ValueRange({ii1, jj1, kk1}), ValueRange({zero, zero, zero}),
        ValueRange({I, J, K}), ValueRange({}),
        [&](ImplicitLocOpBuilder &lb, ValueRange args) {
          ValueRange indices = lb.create<KrnlGetInductionVariableValueOp>(
                                     ValueRange{ii1, jj1, kk1})
                                   .getResults();
          Value i1(indices[0]), j1(indices[1]), k1(indices[2]);
          lb.create<KrnlMatMulOp>(A, ValueRange{zero, zero}, B,
              ValueRange{zero, zero}, C, ValueRange{zero, zero},
              ValueRange{ii2, jj2, kk2}, i1, j1, k1, I, J, K,
              ArrayRef<int64_t>{iRegTile, jRegTile, kRegTile},
              ArrayRef<int64_t>{}, ArrayRef<int64_t>{}, ArrayRef<int64_t>{},
              true, true, false);
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
    // IndexExprScope outerScope(rewriter, shapeHelper.scope);

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
