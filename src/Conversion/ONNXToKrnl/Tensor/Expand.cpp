/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Expand.cpp - Lowering Expand Op----------------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Expand Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXExpandOpLowering : public ConversionPattern {
  ONNXExpandOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXExpandOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get shape.
    ONNXExpandOpAdaptor operandAdaptor(operands);
    ONNXExpandOp expandOp = llvm::dyn_cast<ONNXExpandOp>(op);
    Value input = operandAdaptor.input();
    Location loc = op->getLoc();
    ONNXExpandOpShapeHelper shapeHelper(&expandOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Failed to compute shape");

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    int64_t outputRank = outputMemRefType.getRank();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Iterate over the output values.
    KrnlBuilder createKrnl(rewriter, loc);
    ValueRange outputLoopDef = createKrnl.defineLoops(outputRank);
    LiteralIndexExpr zero(0);
    SmallVector<IndexExpr, 4> lbs(outputRank, zero);
    createKrnl.iterateIE(outputLoopDef, outputLoopDef, lbs,
        shapeHelper.dimsForOutput(0),
        [&](KrnlBuilder &createKrnl, ValueRange outputLoopInd) {
          IndexExprScope outputScope(createKrnl, shapeHelper.scope);
          SmallVector<IndexExpr, 4> outputLoopIndices, lhsAccessExprs;
          getIndexExprList<DimIndexExpr>(outputLoopInd, outputLoopIndices);
          LogicalResult res = shapeHelper.GetAccessExprs(
              input, 0, outputLoopIndices, lhsAccessExprs);
          assert(succeeded(res));
          Value val = createKrnl.loadIE(input, lhsAccessExprs);
          createKrnl.store(val, alloc, outputLoopInd);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXExpandOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXExpandOpLowering>(ctx);
}
