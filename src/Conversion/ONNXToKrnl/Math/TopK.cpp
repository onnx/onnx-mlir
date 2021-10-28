/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- TopK.cpp - TopK Op ---------------------------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

struct ONNXTopKOpLowering : public ConversionPattern {
  ONNXTopKOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTopKOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXTopKOp topkOp = llvm::dyn_cast<ONNXTopKOp>(op);
    ONNXTopKOpAdaptor operandAdaptor(operands);

    KrnlBuilder createKrnl(rewriter, loc);
    MathBuilder createMath(createKrnl);
    IndexExprScope scope(createKrnl);

    MemRefType resMemRefType = convertToMemRefType(*op->result_type_begin());
    auto resElementType = resMemRefType.getElementType();
    Type i64Type = rewriter.getI64Type();
    Type indexType = rewriter.getIndexType();

    // Op's Operands.
    Value X = operandAdaptor.X();
    Value K =
        getOptionalScalarValue(rewriter, loc, operandAdaptor.K(), i64Type, 0);
    SymbolIndexExpr KIE(K);

    // Op's Attributes.
    int64_t rank = resMemRefType.getRank();
    int64_t axis = topkOp.axis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);
    bool ascendingMode = topkOp.largest() != 1;
    // Accoring to ONNX TopK: 'If "sorted" is 0, order of returned 'Values' and
    // 'Indices' are undefined'.
    // In this case, we still return sorted values and indices to make them
    // deterministic. So not used this attribute.
    // bool sortedMode = topkOp.sorted() == 1;

    MemRefBoundsIndexCapture XBounds(X);
    SmallVector<IndexExpr> zeroDims(rank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> XDims, resDims;
    XBounds.getDimList(XDims);
    for (int64_t i = 0; i < rank; ++i)
      if (i == axis)
        resDims.emplace_back(KIE);
      else
        resDims.emplace_back(XDims[i]);

    // Insert an allocation and deallocation for the results of this operation.
    bool insertDealloc = checkInsertDealloc(op, /*resultIndex=*/0);
    Value resMemRef = insertAllocAndDeallocSimple(
        rewriter, op, resMemRefType, loc, resDims, insertDealloc);
    insertDealloc = checkInsertDealloc(op, /*resultIndex=*/1);
    Value resIndexMemRef = insertAllocAndDeallocSimple(rewriter, op,
        MemRefType::get(resMemRefType.getShape(), i64Type), loc, resDims,
        insertDealloc);

    // Compute argsort of X along axis.
    Value argsort =
        emitArgSort(rewriter, loc, X, axis, /*ascending=*/ascendingMode);

    // Produce the final result.
    ValueRange loopDef = createKrnl.defineLoops(rank);
    createKrnl.iterateIE(loopDef, loopDef, zeroDims, resDims,
        [&](KrnlBuilder &createKrnl, ValueRange resLoopInd) {
          Value resInd = createKrnl.load(argsort, resLoopInd);
          SmallVector<Value> resIndexLoopInd(resLoopInd);
          resIndexLoopInd[axis] = resInd;
          // Store value.
          Value val = createKrnl.load(X, resIndexLoopInd);
          createKrnl.store(val, resMemRef, resLoopInd);
          // Store index.
          Value resIndI64 =
              rewriter.create<arith::IndexCastOp>(loc, i64Type, resInd);
          createKrnl.store(resIndI64, resIndexMemRef, resLoopInd);
        });

    rewriter.replaceOp(op, {resMemRef, resIndexMemRef});
    return success();
  }
};

void populateLoweringONNXTopKOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTopKOpLowering>(ctx);
}
