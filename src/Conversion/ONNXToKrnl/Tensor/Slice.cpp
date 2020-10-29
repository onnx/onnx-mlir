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
    for (int ii = 0; ii < outputRank; ++ii)
      outputLoops.pushBounds(
          shapeHelper.context, 0, shapeHelper.outputDims[ii]);
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    IndexExprContext childContext(shapeHelper.context);
#if 0
    printf("\nalex, make a test for affine\n");
    Value ind0 = outputLoops.getInductionVar(0);
    IndexExpr i1 = outerloopContex.CreateSymbolIndex(ind0);
    IndexExpr t1;
    t1.Sub(i1, i1);
    t1.DebugPrint("index loop i -i");
#endif

    // Proceed with the load data["i * step + start} for all dim].
    Value loadVal;
    SmallVector<Value, 4> loadIndices;
    bool loadIsAffine = true;
    for (int ii = 0; ii < outputRank; ++ii) {
      Value loopVal = outputLoops.getInductionVar(ii);
      IndexExpr loopIndex, start, step, actualIndex;
      loopIndex = childContext.createDimIndex(loopVal);
      start = childContext.createSymbolIndexFromParentContext(
          shapeHelper.starts[ii]);
      step = childContext.createSymbolIndexFromParentContext(
          shapeHelper.steps[ii]);
      loopIndex.DebugPrint("loop index");
      step.DebugPrint("  steps");
      start.DebugPrint("  start");
      actualIndex.Mult(step, loopIndex).IncBy(start);
      loadIndices.emplace_back(actualIndex.getValue());
      if (!actualIndex.isAffine())
        loadIsAffine = false;
    }
    // Load data.
    if (loadIsAffine) {
      loadVal = rewriter.create<AffineLoadOp>(
          loc, operandAdaptor.data(), loadIndices);
    } else {
      loadVal =
          rewriter.create<LoadOp>(loc, operandAdaptor.data(), loadIndices);
    }
    // Store data
    SmallVector<Value, 4> storeIndices;
    for (int ii = 0; ii < outputRank; ++ii) {
      storeIndices.emplace_back(outputLoops.getInductionVar(ii));
    }
    rewriter.create<AffineStoreOp>(loc, loadVal, alloc, storeIndices);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSliceOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSliceOpLowering>(ctx);
}
