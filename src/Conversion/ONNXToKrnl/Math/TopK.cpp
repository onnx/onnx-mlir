/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- TopK.cpp - TopK Op ---------------------------===//
//
// Copyright 2021-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX TopK operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXTopKOpLowering : public ConversionPattern {
  ONNXTopKOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXTopKOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXTopKOp topkOp = llvm::dyn_cast<ONNXTopKOp>(op);
    ONNXTopKOpAdaptor operandAdaptor(operands);
    Value X = operandAdaptor.X();

    // Builders.
    KrnlBuilder createKrnl(rewriter, loc);

    // Common types.
    MemRefType resMemRefType = convertToMemRefType(*op->result_type_begin());
    Type i64Type = rewriter.getI64Type();

    // Op's Attributes.
    int64_t rank = resMemRefType.getRank();
    int64_t axis = topkOp.axis();
    axis = axis < 0 ? axis + rank : axis;
    assert(axis >= 0 && axis < rank && "axis is out of bound");
    bool ascendingMode = topkOp.largest() != 1;
    // Accoring to ONNX TopK: 'If "sorted" is 0, order of returned 'Values' and
    // 'Indices' are undefined'.
    // In this case, we still return sorted values and indices to make them
    // deterministic. So this attribute is not used.
    // bool sortedMode = topkOp.sorted() == 1;

    // Compute the output's dimension sizes.
    ONNXTopKOpShapeHelper shapeHelper(&topkOp, &rewriter,
        getDenseElementAttributeFromConstantValue,
        krnl::loadDenseElementArrayValueAtIndex);
    auto shapeComputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapeComputed) && "Could not compute output shape");
    auto resDims = shapeHelper.dimsForOutput();

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
    SmallVector<IndexExpr> zeroDims(rank, LiteralIndexExpr(0));
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

void populateLoweringONNXTopKOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXTopKOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
