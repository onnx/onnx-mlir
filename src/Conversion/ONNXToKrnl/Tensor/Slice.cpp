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

    IndexExprContainer container(&rewriter, sliceOp.getLoc());
    SmallVector<IndexExpr, 4> startsIEV;
    SmallVector<IndexExpr, 4> stepsIEV;
    SmallVector<IndexExpr, 4> endsIEV;
    SmallVector<IndexExpr, 4> outputDimsIEV;
    if (failed(HandleSliceOpParams(&sliceOp, operandAdaptor, container,
            startsIEV, endsIEV, stepsIEV, outputDimsIEV))) {
      // Failed to slice parameters.
      return sliceOp.emitError("failure to get Slice parameters");
    }
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputMemRefShape = outputMemRefType.getShape();
    int64_t outputRank = outputMemRefShape.size();
    assert(outputRank == outputDimsIEV.size());
// Insert an allocation and deallocation for the output of this operation.
#if 1
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, container, op, outputMemRefType, loc, outputDimsIEV);
#else
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(outputMemRefType))
      alloc =
          insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);
    else {
      SmallVector<Value, 2> allocOperands;
      for (int i = 0; i < outputRank; ++i) {
        if (outputMemRefShape[i] < 0) {
          // have dyn shape
          allocOperands.emplace_back(outputDimsIEV[i].GetValue(container));
        }
      }
      AllocOp allocOp =
          rewriter.create<AllocOp>(loc, outputMemRefType, allocOperands);
      if (insertDealloc) {
        auto *parentBlock = allocOp.getOperation()->getBlock();
        auto dealloc = rewriter.create<DeallocOp>(loc, allocOp);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
      alloc = allocOp;
    }
#endif

    BuildKrnlLoop outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    for (int ii = 0; ii < outputRank; ++ii)
      outputLoops.pushBounds(container, 0, outputDimsIEV[ii]);
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // proceed with the load data["i * step + start} for all dim]
    Value loadVal;
    SmallVector<Value, 4> loadIndices;
    bool loadIsAffine = true;
    for (int ii = 0; ii < outputRank; ++ii) {
      Value loopIndex = outputLoops.getInductionVar(ii);
      if (stepsIEV[ii].IsIntLit() && startsIEV[ii].IsAffine()) {
        // affine, can reuse the same affine container
        IndexExpr loopIndexIE, multIE, addIE;
        loopIndexIE.InitAsDim(container, loopIndex);
        multIE.Mult(container, stepsIEV[ii], loopIndexIE);
        addIE.Add(container, multIE, startsIEV[ii]);
        loadIndices.emplace_back(addIE.GetValue(container));
      } else {
        IndexExprContainer newContainer(&rewriter, loc);
        IndexExpr loopIndexIE, stepIE, startIE, multIE, addIE;
        loopIndexIE.InitAsDim(container, loopIndex);
        startIE.InitAsSymbol(newContainer, startsIEV[ii].GetValue(container));
        stepIE.InitAsSymbol(newContainer, stepsIEV[ii].GetValue(container));
        multIE.Mult(newContainer, stepIE, loopIndexIE);
        addIE.Add(newContainer, multIE, startIE);
        loadIndices.emplace_back(addIE.GetValue(newContainer));
        if (!addIE.IsAffine())
          loadIsAffine = false;
      }
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
