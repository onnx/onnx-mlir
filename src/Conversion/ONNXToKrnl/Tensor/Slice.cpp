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
    printf("alex: start handle slice\n");
    if (failed(HandleSliceOpParams(&sliceOp, operandAdaptor, container,
            startsIEV, endsIEV, stepsIEV, outputDimsIEV))) {
      // Failed to slice parameters.
    printf("alex: stop with failure handle slice\n");
      return sliceOp.emitError("failure to get Slice parameters");
    }
    printf("alex: stop with success handle slice\n");
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
      loopIndexIE.InitAsDim(loopIndex);
      if (stepsIEV[ii].IsIntLit() && startsIEV[ii].IsAffine()) {
        // affine, can reuse the same affine container
        multIE.Mult(container, stepsIEV[ii], loopIndexIE);
        addIE.Add(container, multIE, startsIEV[ii]);
        loadIndices.emplace_back(addIE.GetValue(container));
      } else {
        loadIsAffine = false;
        IndexExprContainer newContainer(&rewriter, loc);
        IndexExpr stepIE, startIE;
        startIE.InitAsSymbol(startsIEV[ii].GetValue(container));
        stepIE.InitAsSymbol(stepsIEV[ii].GetValue(container));
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

#if 0     
    // get input operands, shapes, and rank
    Value data = operandAdaptor.data();
    auto dataShape = data.getType().cast<MemRefType>().getShape();
    int64_t dataRank = dataShape.size();
    Value indices = operandAdaptor.indices();
    auto indicesShape = indices.getType().cast<MemRefType>().getShape();
    int64_t indicesRank = indicesShape.size();
    int64_t axisIndex = SliceOp.axis();
    // get output info
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputMemRefShape = outputMemRefType.getShape();
    int64_t outputRank = outputMemRefShape.size();
    /*
      The pattern that we are using is that of numpy.take.

      Ni, Nk = data.shape[:axis], data.shape[axis+1:]
      Nj = indices.shape
      for ii in ndindex(Ni):
        for jj in ndindex(Nj):
          for kk in ndindex(Nk):
            out[ii + jj + kk] = data[ii + (indices[jj],) + kk]
    */
    // Define loops and iteration trip counts (equivalent to size of output)
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, outputRank);
    KrnlIterateOperandPack pack(rewriter, originalLoops);
    int iIndexStart = 0;
    for (int ii = 0; ii < axisIndex; ++ii)
      addDimensionToPack(rewriter, loc, pack, data, ii);
    // Then iterates over the Nj (indices matrix), jj indices in above algo.
    int jIndexStart = iIndexStart + axisIndex;
    for (int jj = 0; jj < indicesRank; ++jj)
      addDimensionToPack(rewriter, loc, pack, indices, jj);
    // Finally iterates over the Nk (data after axis), kk indices in above algo.
    int kIndexStart = jIndexStart + indicesRank - (axisIndex + 1);
    for (int kk = axisIndex + 1; kk < dataRank; ++kk)
      addDimensionToPack(rewriter, loc, pack, data, kk);
    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(outputMemRefType))
      alloc =
          insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);
    else
      return emitError(loc, "unsupported dynamic dimensions");

    // Get the size of the "axis"th dimension of data.
    auto zeroConst = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    auto axisIndexConst = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIndexType(), axisIndex));
    auto sizeAxisVal = rewriter.create<KrnlDimOp>(
        loc, rewriter.getIndexType(), data, axisIndexConst);

    // Create the loops
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the just generated loops.
    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operations.
    // Read first the indices[jj] into rawIndexVal.
    SmallVector<Value, 4> indicesMemRefVal;
    for (int j = 0; j < indicesRank; ++j)
      indicesMemRefVal.emplace_back(
          iterationBlock.getArguments()[jIndexStart + j]);
    auto indexValInteger =
        rewriter.create<AffineLoadOp>(loc, indices, indicesMemRefVal);
    auto rawIndexVal = rewriter.create<IndexCastOp>(
        loc, indexValInteger, rewriter.getIndexType());
    // When raw index value is negative, must add array dimension size to it.
    auto negativeIndexVal =
        rewriter.create<AddIOp>(loc, rawIndexVal, sizeAxisVal);
    // Select value for non-negative or negative case.
    auto isNegative = rewriter.create<CmpIOp>(
        loc, CmpIPredicate::slt, rawIndexVal, zeroConst);
    auto indexVal = rewriter.create<SelectOp>(
        loc, isNegative, negativeIndexVal, rawIndexVal);

    // Then read input data into DataVal: first add ii's.
    SmallVector<Value, 4> dataMemRefVal;
    for (int i = 0; i < axisIndex; ++i)
      dataMemRefVal.emplace_back(
          iterationBlock.getArguments()[iIndexStart + i]);
    // Then add indices[jj] (indexVal).
    dataMemRefVal.emplace_back(indexVal);
    // Then add kk's.
    for (int k = axisIndex + 1; k < dataRank; ++k)
      dataMemRefVal.emplace_back(
          iterationBlock.getArguments()[kIndexStart + k]);
    auto dataVal = rewriter.create<LoadOp>(loc, data, dataMemRefVal);

    // Then store the value in the output.
    SmallVector<Value, 4> outputMemRefVal;
    for (int n = 0; n < iterationBlock.getArguments().size(); ++n)
      outputMemRefVal.emplace_back(iterationBlock.getArguments()[n]);
    rewriter.create<AffineStoreOp>(loc, dataVal, alloc, outputMemRefVal);

    rewriter.replaceOp(op, alloc);

#endif

    return success();
  }
};

void populateLoweringONNXSliceOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSliceOpLowering>(ctx);
}
