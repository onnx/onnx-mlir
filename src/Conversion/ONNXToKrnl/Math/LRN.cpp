
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

    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    MemRefType scalarIndexType = MemRefType::get({}, rewriter.getIndexType(), {}, 0);
    Value cAlloc = rewriter.create<AllocOp>(loc, scalarIndexType);
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    Value sumAlloc = rewriter.create<AllocOp>(loc, scalarMemRefType);

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
    IndexExprContext childContext(shapeHelper.context);
    
    // Compute the lower bound and upper bound for square_sum.
    const int loopIndexForC = 1;
    Value cValue = outputLoops.getInductionVar(loopIndexForC);
    IndexExpr cIE = childContext.createLoopInductionIndex(cValue);
    IndexExpr sizeIE = childContext.createDimIndexFromShapedType(input, loopIndexForC);
    
    SmallVector<IndexExpr, 2> lbMaxList;
    lbMaxList.emplace_back(childContext.createLiteralIndex(0));
    //lbMaxList.emplace_back(cIE-(sizeIE-childContext.createLiteralIndex(1)).floorDiv(childContext.createLiteralIndex(2)));
    lbMaxList.emplace_back(cIE-(sizeIE-childContext.createLiteralIndex(1)));
    //IndexExpr lbIE = IndexExpr::max(lbMaxList);

    SmallVector<IndexExpr, 2> ubMinList;
    ubMinList.emplace_back(sizeIE-1);
    ubMinList.emplace_back(cIE-childContext.createLiteralIndex(1)+(sizeIE-childContext.createLiteralIndex(1)).ceilDiv(childContext.createLiteralIndex(2)));

    IndexExpr ubIE = cIE-childContext.createLiteralIndex(1)+(sizeIE-childContext.createLiteralIndex(1)).floorDiv(childContext.createLiteralIndex(2));
    
    rewriter.create<StoreOp>(loc, ubIE.getValue(), cAlloc, ArrayRef<Value>{});
    Value cLoad = rewriter.create<LoadOp>(loc, cAlloc, ArrayRef<Value>{});

    // Initialize sum
    rewriter.create<StoreOp>(loc, emitConstantOp(rewriter, loc, elementType, 0), sumAlloc, ArrayRef<Value>{});

    // Create the sum reduction loop
    BuildKrnlLoop sumLoops(rewriter, loc, 1);
    sumLoops.createDefineOp();
    //sumLoops.pushBounds(lbIE.getValue(), ubIE.getValue());
    // Error sumLoops.pushBounds(0, cValue);
    sumLoops.pushBounds(0, cIE);
    sumLoops.createIterateOp();
    auto outputLoopBody = rewriter.saveInsertionPoint(); 
    rewriter.setInsertionPointToStart(sumLoops.getIterateBlock());
    
    // Compute quare-sum value
    SmallVector<Value, 4> loadIndices;
    for (int i = 0; i < outputRank; i++) {
      if (i == loopIndexForC) {
        Value loopVal = outputLoops.getInductionVar(i);
        //loadIndices.emplace_back(loopVal);
        loadIndices.emplace_back(cLoad);
      } else {
        Value loopVal = outputLoops.getInductionVar(i);
        loadIndices.emplace_back(loopVal);
      }
    }

    Value loadVal = rewriter.create<LoadOp>(loc, input, loadIndices);
    Value squareVal = rewriter.create<MulFOp>(loc, loadVal, loadVal);
  
    Value sumValue = rewriter.create<LoadOp>(loc, sumAlloc, ArrayRef<Value>{});
    sumValue = rewriter.create<AddFOp>(loc, sumValue, squareVal);
    rewriter.create<StoreOp>(loc, sumValue, sumAlloc, ArrayRef<Value>{});

    rewriter.restoreInsertionPoint(outputLoopBody);
    SmallVector<Value, 4> storeIndices;
    for (int i = 0; i < outputRank; ++i) {
      storeIndices.emplace_back(outputLoops.getInductionVar(i));
    }
    sumValue = rewriter.create<LoadOp>(loc, sumAlloc, ArrayRef<Value>{});
    rewriter.create<StoreOp>(loc, sumValue, alloc, storeIndices);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXLRNOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXLRNOpLowering>(ctx);
}
