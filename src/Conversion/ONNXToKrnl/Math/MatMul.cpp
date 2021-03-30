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

#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"

using namespace mlir;

struct ONNXMatMulOpLowering : public ConversionPattern {
  ONNXMatMulOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXMatMulOp::getOperationName(), 1, ctx) {}

  // Handle the generic cases, including when there are broadcasts.
  void replaceGenericMatmul(ONNXMatMulOp &matMulOp,
      ONNXMatMulOpAdaptor &operandAdaptor, Type elementType,
      ONNXMatMulOpShapeHelper &shapeHelper, Value alloc, Value zero,
      ConversionPatternRewriter &rewriter, Location loc) const {

    // Non-reduction loop iterations: output-rank.
    int outerloopNum = shapeHelper.dimsForOutput(0).size();
    BuildKrnlLoop outputLoops(rewriter, loc, outerloopNum);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // Access function for the output, and set it to zero.
    SmallVector<IndexExpr, 4> resAccessFct;
    getIndexExprList<DimIndexExpr>(
        outputLoops.getAllInductionVar(), resAccessFct);
    // Insert res[...] = 0.
    // Create a local reduction value for res[...].
    Value reductionVal =
        rewriter.create<AllocaOp>(loc, MemRefType::get({}, elementType));
    rewriter.create<KrnlStoreOp>(loc, zero, reductionVal, ArrayRef<Value>{});

    // Create the inner reduction loop; trip count is last dim of A.
    BuildKrnlLoop innerLoops(rewriter, loc, 1);
    innerLoops.createDefineOp();
    int aRank = shapeHelper.aDims.size();
    int bRank = aRank; // Add for better readability.
    innerLoops.pushBounds(0, shapeHelper.aDims[aRank - 1]);
    innerLoops.createIterateOp();

    // Now start writing code inside the inner loop: get A & B access functions.
    auto ipOuterLoopRegion = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(innerLoops.getIterateBlock());

    DimIndexExpr k(innerLoops.getInductionVar(0));
    SmallVector<IndexExpr, 4> aAccessFct, bAccessFct;
    for (int i = 0; i < aRank; ++i) {
      // Add index if dim is not a padded dimension.
      if (!shapeHelper.aPadDims[i]) {
        // For A, reduction index is last
        if (i == aRank - 1) {
          aAccessFct.emplace_back(k);
        } else {
          aAccessFct.emplace_back(resAccessFct[i]);
        }
      }
      if (!shapeHelper.bPadDims[i]) {
        // For B, reduction index is second to last.
        if (i == bRank - 2) {
          bAccessFct.emplace_back(k);
        } else if (i == outerloopNum) {
          // When the rank of A 1D, then the output lost one dimension.
          // E,g, (5) x (10, 5, 4) -> padded (1, 5) x (10, 5, 4) = (10, 1, 4).
          // But we drop the "1" so its really (10, 4). When processing the
          // last dim of the reduction (i=2 here), we would normally access
          // output[2] but it does not exist, because we lost a dim in the
          // output due to 1D A.
          bAccessFct.emplace_back(resAccessFct[i - 1]);
        } else {
          bAccessFct.emplace_back(resAccessFct[i]);
        }
      }
    }

    // Add mat mul operation.
    Value loadedA = krnl_load(operandAdaptor.A(), aAccessFct);
    Value loadedB = krnl_load(operandAdaptor.B(), bAccessFct);
    Value loadedY =
        rewriter.create<KrnlLoadOp>(loc, reductionVal, ArrayRef<Value>{});
    Value AB = rewriter.create<MulFOp>(loc, loadedA, loadedB);
    Value accumulated = rewriter.create<AddFOp>(loc, loadedY, AB);
    rewriter.create<KrnlStoreOp>(
        loc, accumulated, reductionVal, ArrayRef<Value>{});

    rewriter.restoreInsertionPoint(ipOuterLoopRegion);
    accumulated =
        rewriter.create<KrnlLoadOp>(loc, reductionVal, ArrayRef<Value>{});
    krnl_store(accumulated, alloc, resAccessFct);
  }

  // Handle the cases with 2x2 matrices both for A, B, and C without broadcast.
  // Implementation here uses the efficient 1d tiling plus kernel substitution.
  void replace2x2Matmul2D(ONNXMatMulOp &matMulOp,
      ONNXMatMulOpAdaptor &operandAdaptor, Type elementType,
      ONNXMatMulOpShapeHelper &shapeHelper, Value alloc, Value zeroVal,
      ConversionPatternRewriter &rewriter, Location loc) const {

    using namespace mlir::edsc;
    using namespace mlir::edsc::ops;
    using namespace mlir::edsc::intrinsics;

    // Define scopes
    ScopedContext scope(rewriter, loc);
    IndexExprScope indexScope(rewriter, loc);

    // Prepare: loop bounds and zero
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(alloc);
    MemRefBoundIndexCapture ABounds(A), CBounds(C);
    IndexExpr I = CBounds.getDim(0);
    IndexExpr J = CBounds.getDim(1);
    IndexExpr K = ABounds.getDim(1);
    LiteralIndexExpr zero(0);

    // Initialize alloc/C to zero.
    ValueRange zLoop = krnl_define_loop(2);
    Value zi(zLoop[0]), zj(zLoop[1]);
    krnl_iterate({zi, zj}, {zero, zero}, {I, J}, {}, [&](ArrayRef<Value> args) {
      ValueRange indices = krnl_get_induction_var_value({zi, zj});
      DimIndexExpr zii(indices[0]), zjj(indices[1]);
      SmallVector<IndexExpr> storeIndices({zii, zjj});
      krnl_store(zeroVal, alloc, storeIndices);
    });

    // Compute.
    // Define blocking, with simdization along the j axis.
    const int64_t iRegTile(4), jRegTile(8), kRegTile(2);
    // I, J, K loop.
    ValueRange origLoop = krnl_define_loop(3);
    Value i(origLoop[0]), j(origLoop[1]), k(origLoop[2]);
    // Define blocked loop and permute.
    ValueRange iRegBlock = krnl_block(i, iRegTile);
    Value iiB(iRegBlock[0]), iiL(iRegBlock[1]);
    ValueRange jRegBlock = krnl_block(j, jRegTile);
    Value jjB(jRegBlock[0]), jjL(jRegBlock[1]);
    ValueRange kRegBlock = krnl_block(k, kRegTile);
    Value kkB(kRegBlock[0]), kkL(kRegBlock[1]);
    krnl_permute({iiB, iiL, jjB, jjL, kkB, kkL}, {0, 3, 1, 4, 2, 5});

    krnl_iterate({i, j, k}, {iiB, jjB, kkB}, {zero, zero, zero}, {I, J, K}, {},
        [&](ArrayRef<Value> args) {
          ValueRange indices = krnl_get_induction_var_value({iiB, jjB, kkB});
          DimIndexExpr ii(indices[0]), jj(indices[1]), kk(indices[2]);
          SmallVector<int64_t, 4> computeTile;
          computeTile = {iRegTile, jRegTile, kRegTile};
          krnl_matmul({iiL, jjL, kkL}, A, B, C, {ii, jj, kk}, {I, J, K},
              computeTile, {}, {}, {}, true, false, false);
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
    ONNXMatMulOpShapeHelper shapeHelper(&matMulOp, &rewriter);
    LogicalResult shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));
    IndexExprScope outerScope(shapeHelper.scope);

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = outputMemRefType.getElementType();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Get the constants: zero.
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);

    if (shapeHelper.aDims.size() == 2 && shapeHelper.bDims.size() == 2 &&
        IndexExpr::isLiteral(shapeHelper.aDims) &&
        IndexExpr::isLiteral(shapeHelper.bDims)) {
      replace2x2Matmul2D(matMulOp, operandAdaptor, elementType, shapeHelper,
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
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLowering>(ctx);
}
