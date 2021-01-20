
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
    int64_t outputRank = outputMemRefShape.size();

    Value input = operandAdaptor.X();

    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    BuildKrnlLoop outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    SmallVector<Value, 4> loadIndices;
    bool isAffineLoad = true;

    for (int i = 0; i < outputRank; i++) {
      Value loopVal = outputLoops.getInductionVar(i);
      loadIndices.emplace_back(loopVal);
    }

    Value loadVal;
    loadVal = rewriter.create<AffineLoadOp>(loc, input, loadIndices);

    SmallVector<Value, 4> storeIndices;
    for (int i = 0; i < outputRank; ++i) {
      storeIndices.emplace_back(outputLoops.getInductionVar(i));
    }
    rewriter.create<AffineStoreOp>(loc, loadVal, alloc, storeIndices);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXLRNOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXLRNOpLowering>(ctx);
}
