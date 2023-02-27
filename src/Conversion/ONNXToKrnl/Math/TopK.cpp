/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- TopK.cpp - TopK Op ---------------------------===//
//
// Copyright 2021-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX TopK operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXTopKOpLowering : public ConversionPattern {
  ONNXTopKOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXTopKOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXTopKOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Value X = operandAdaptor.getX();

    // Builders.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType resMemRefType = convertedType.cast<MemRefType>();

    // Common types.
    Type i64Type = rewriter.getI64Type();

    // Op's Attributes.
    int64_t rank = resMemRefType.getRank();
    int64_t axis = operandAdaptor.getAxis();
    axis = axis < 0 ? axis + rank : axis;
    assert(axis >= 0 && axis < rank && "axis is out of bound");
    bool ascendingMode = operandAdaptor.getLargest() != 1;
    // According to ONNX TopK: 'If "sorted" is 0, order of returned 'Values' and
    // 'Indices' are undefined'.
    // In this case, we still return sorted values and indices to make them
    // deterministic. So this attribute is not used.
    // bool sortedMode = TopKOp.sorted() == 1;

    // Compute the output's dimension sizes.
    ONNXTopKOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr resDims = shapeHelper.getOutputDims();

    // Insert an allocation and deallocation for the results of this operation.
    Value resMemRef = create.mem.alignedAlloc(resMemRefType, resDims);
    Value resIndexMemRef = create.mem.alignedAlloc(
        MemRefType::get(resMemRefType.getShape(), i64Type), resDims);

    // Compute argSort of X along axis.
    Value argSort = emitArgSort(rewriter, loc, X, axis,
        /*ascending=*/ascendingMode);

    // Produce the final result.
    SmallVector<IndexExpr> zeroDims(rank, LiteralIndexExpr(0));
    ValueRange loopDef = create.krnl.defineLoops(rank);
    create.krnl.iterateIE(loopDef, loopDef, zeroDims, resDims,
        [&](KrnlBuilder &createKrnl, ValueRange resLoopInd) {
          Value resInd = createKrnl.load(argSort, resLoopInd);
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
