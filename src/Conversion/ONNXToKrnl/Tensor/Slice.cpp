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
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(outputMemRefType))
      alloc =
          insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);
    else {
      llvm_unreachable("not there yet");
    }

    BuildKrnlLoop outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    for (int ii = 0; ii < outputRank; ++ii) {
      if (outputDimsIEV[ii].IsIntLit()) {
        outputLoops.pushBounds(0, outputDimsIEV[ii].GetIntLit());
      } else {
        outputLoops.pushBounds(0, outputDimsIEV[ii].GetValue(container));
      }
    }
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // proceed with the load data["i * step + start} for all dim]
    Value loadVal;
    SmallVector<Value, 4> loadIndices;
    bool loadIsAffine = true;
    for (int ii = 0; ii < outputRank; ++ii) {
      Value loopIndex = outputLoops.getInductionVar(ii);
      IndexExpr loopIndexIE, multIE, addIE;
      loopIndexIE.InitAsDim(container, loopIndex);
      if (stepsIEV[ii].IsIntLit() && startsIEV[ii].IsAffine()) {
        // affine, can reuse the same affine container
        multIE.Mult(container, stepsIEV[ii], loopIndexIE);
        addIE.Add(container, multIE, startsIEV[ii]);
        loadIndices.emplace_back(addIE.GetValue(container));
      } else {
        loadIsAffine = false;
        IndexExprContainer newContainer(&rewriter, loc);
        IndexExpr stepIE, startIE;
        startIE.InitAsSymbol(newContainer, startsIEV[ii].GetValue(container));
        stepIE.InitAsSymbol(newContainer, stepsIEV[ii].GetValue(container));
        multIE.Mult(newContainer, stepIE, loopIndexIE);
        addIE.Add(newContainer, multIE, startIE);
        loadIndices.emplace_back(addIE.GetValue(newContainer));
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
