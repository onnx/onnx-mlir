//===----------------Slice.cpp - Lowering Slice Op----------------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Slice Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXSliceOpLowering : public ConversionPattern {
  ONNXSliceOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSliceOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSliceOpAdaptor operandAdaptor(operands);
    ONNXSliceOp sliceOp = llvm::cast<ONNXSliceOp>(op);
    auto loc = op->getLoc();

    ONNXSliceOpShapeHelper shapeHelper(&sliceOp, &rewriter);
    if (failed(shapeHelper.Compute(operandAdaptor)))
      return op->emitError("Failed to scan Silce parameters successfully");

    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    int64_t outputRank = outputMemRefType.getShape().size();
    assert(outputRank == shapeHelper.outputDims.size());
    // Insert an allocation and deallocation for the output of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.outputDims);

    BuildKrnlLoop outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.outputDims);
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    IndexExprContext childContext(shapeHelper.context);

#if 0
    printf("\nalex, make a test for affine\n");
    Value ind0 = outputLoops.getInductionVar(0);
    IndexExpr i1 = outerloopContex.CreateSymbolIndex(ind0);
    IndexExpr t1;
    t1.Sub(i1, i1);
    t1.debugPrint("index loop i -i");
#endif

    // Compute indices for the load and store op.
    // Load: "i * step + start" for all dim.
    // Store: "i".
    SmallVector<IndexExpr, 4> loadIndices;
    SmallVector<IndexExpr, 4> storeIndices;
    for (int ii = 0; ii < outputRank; ++ii) {
      Value loopVal = outputLoops.getInductionVar(ii);
      IndexExpr loopIndex = childContext.createLoopIterIndex(loopVal);
      IndexExpr start = childContext.createSymbolIndexFromParentContext(
          shapeHelper.starts[ii]);
      IndexExpr step = childContext.createSymbolIndexFromParentContext(
          shapeHelper.steps[ii]);
      loopIndex.debugPrint("loop index");
      step.debugPrint("  steps");
      start.debugPrint("  start");
      loadIndices.emplace_back((step * loopIndex) + start);
      storeIndices.emplace_back(loopIndex);
    }
    // Load data and store in alloc data.
    Value loadVal =
        childContext.createLoadOp(operandAdaptor.data(), loadIndices);
    childContext.createStoreOp(loadVal, alloc, storeIndices);

    rewriter.replaceOp(op, alloc);
    printf("done alex\n\n");
    return success();
  }
};

void populateLoweringONNXSliceOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSliceOpLowering>(ctx);
}
