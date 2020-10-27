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

    IndexExprContext outerloopContex(&rewriter, sliceOp.getLoc());
    SmallVector<IndexExpr, 4> starts;
    SmallVector<IndexExpr, 4> steps;
    SmallVector<IndexExpr, 4> ends;
    SmallVector<IndexExpr, 4> outputDims;
    if (failed(HandleSliceOpParams(&sliceOp, operandAdaptor, outerloopContex,
            starts, ends, steps, outputDims))) {
      // Failed to slice parameters.
      return sliceOp.emitError("failure to get Slice parameters");
    }
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputMemRefShape = outputMemRefType.getShape();
    int64_t outputRank = outputMemRefShape.size();
    assert(outputRank == outputDims.size());
    // Insert an allocation and deallocation for the output of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, outputDims);

    BuildKrnlLoop outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    for (int ii = 0; ii < outputRank; ++ii)
      outputLoops.pushBounds(outerloopContex, 0, outputDims[ii]);
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    IndexExprContext childContext(outerloopContex);

    // proceed with the load data["i * step + start} for all dim]
    Value loadVal;
    SmallVector<Value, 4> loadIndices;
    bool loadIsAffine = true;
    for (int ii = 0; ii < outputRank; ++ii) {
      Value loopVal = outputLoops.getInductionVar(ii);
      IndexExpr loopIndex, start, step, actualIndex;
      // Decide here if we can reuse the parent outerloopContex: can do so if
      // start is afine and steps are literals.
      loopIndex = childContext.CreateDimIndex(loopVal);
      start = childContext.CreateSymbolIndexFromParentContext(starts[ii]);
      step = childContext.CreateSymbolIndexFromParentContext(steps[ii]);
      loopIndex.DebugPrint("loop index");
      step.DebugPrint("  steps");
      start.DebugPrint("  start");
      actualIndex.Mult(step, loopIndex).IncBy(start);
      loadIndices.emplace_back(actualIndex.GetValue());
      if (!actualIndex.IsAffine())
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
