/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------LRN.cpp - Lowering LRN Op----------------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX LRN Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXLRNOpLowering : public ConversionPattern {
  ONNXLRNOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXLRNOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXLRNOpAdaptor operandAdaptor(operands);
    ONNXLRNOp lrnOp = llvm::cast<ONNXLRNOp>(op);
    auto loc = op->getLoc();

    ONNXLRNOpShapeHelper shapeHelper(&lrnOp, &rewriter);

    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    (void)shapecomputed;
    assert(!failed(shapecomputed) && "expected to succeed");

    auto resultOperand = lrnOp.Y();
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputMemRefShape = outputMemRefType.getShape();
    auto elementType = outputMemRefType.getElementType();
    int64_t outputRank = outputMemRefShape.size();

    Value input = operandAdaptor.X();
    float biasLit = lrnOp.bias().convertToFloat();
    float alphaLit = lrnOp.alpha().convertToFloat();
    float betaLit = lrnOp.beta().convertToFloat();
    float sizeLit = (float)lrnOp.size();
    auto f32Type = FloatType::getF32(rewriter.getContext());
    Value biasValue = emitConstantOp(rewriter, loc, f32Type, biasLit);
    Value alphaDivSizeValue =
        emitConstantOp(rewriter, loc, f32Type, alphaLit / sizeLit);
    Value betaValue = emitConstantOp(rewriter, loc, f32Type, betaLit);

    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    BuildKrnlLoop outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // Insert computation of square_sum.
    // square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
    // where max(0, c - floor((size - 1) / 2)) <= i
    // and i<= min(C - 1, c + ceil((size - 1) / 2)).

    // Get a child IndexExpr context.
    IndexExprScope childScope(shapeHelper.scope);

    // Compute the lower bound and upper bound for square_sum.
    const int loopIndexForC = 1;
    Value cValue = outputLoops.getInductionVar(loopIndexForC);
    DimIndexExpr cIE(cValue);
    MemRefBoundIndexCapture inputBounds(input);
    DimIndexExpr sizeIE(inputBounds.getDim(loopIndexForC));

    SmallVector<IndexExpr, 2> lbMaxList;
    lbMaxList.emplace_back(LiteralIndexExpr(0));
    lbMaxList.emplace_back(cIE - (sizeIE - 1).floorDiv(LiteralIndexExpr(2)));

    SmallVector<IndexExpr, 2> ubMinList;
    ubMinList.emplace_back(sizeIE);
    ubMinList.emplace_back(cIE + 1 + (sizeIE - 1).ceilDiv(LiteralIndexExpr(2)));

    // Initialize sum
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    Value sumAlloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
    rewriter.create<KrnlStoreOp>(loc,
        emitConstantOp(rewriter, loc, elementType, 0), sumAlloc,
        ArrayRef<Value>{});

    // Create the sum reduction loop
    BuildKrnlLoop sumLoops(rewriter, loc, 1);
    sumLoops.createDefineOp();
    sumLoops.pushBounds(lbMaxList, ubMinList);
    sumLoops.createIterateOp();
    auto outputLoopBody = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(sumLoops.getIterateBlock());

    // Compute quare-sum value
    SmallVector<Value, 4> loadIndices;
    for (int i = 0; i < outputRank; i++) {
      if (i == loopIndexForC) {
        Value loopVal = sumLoops.getInductionVar(0);
        loadIndices.emplace_back(loopVal);
      } else {
        Value loopVal = outputLoops.getInductionVar(i);
        loadIndices.emplace_back(loopVal);
      }
    }

    Value loadVal = rewriter.create<KrnlLoadOp>(loc, input, loadIndices);
    Value squareVal = rewriter.create<MulFOp>(loc, loadVal, loadVal);

    Value sumValue =
        rewriter.create<KrnlLoadOp>(loc, sumAlloc, ArrayRef<Value>{});
    sumValue = rewriter.create<AddFOp>(loc, sumValue, squareVal);
    rewriter.create<KrnlStoreOp>(loc, sumValue, sumAlloc, ArrayRef<Value>{});

    // Compute and store the output
    // y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
    rewriter.restoreInsertionPoint(outputLoopBody);
    SmallVector<Value, 4> storeIndices;
    for (int i = 0; i < outputRank; ++i) {
      storeIndices.emplace_back(outputLoops.getInductionVar(i));
    }
    Value xValue = rewriter.create<KrnlLoadOp>(loc, input, storeIndices);
    sumValue = rewriter.create<KrnlLoadOp>(loc, sumAlloc, ArrayRef<Value>{});
    Value tempValue = rewriter.create<math::PowFOp>(loc,
        rewriter.create<AddFOp>(loc, biasValue,
            rewriter.create<MulFOp>(loc, alphaDivSizeValue, sumValue)),
        betaValue);
    Value resultValue = rewriter.create<DivFOp>(loc, xValue, tempValue);

    rewriter.create<KrnlStoreOp>(loc, resultValue, alloc, storeIndices);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXLRNOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXLRNOpLowering>(ctx);
}
