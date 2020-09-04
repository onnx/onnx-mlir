//===----------------Gather.cpp - Lowering Gather Op----------------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gather Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXGatherOpLowering : public ConversionPattern {
  ONNXGatherOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXGatherOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXGatherOpAdaptor operandAdaptor(operands);
    ONNXGatherOp gatherOp = llvm::dyn_cast<ONNXGatherOp>(op);
    auto loc = op->getLoc();
    printf("hi alex: start\n");
    // get input operands, shapes, and rank
    Value data = operandAdaptor.data();
    auto dataShape = data.getType().cast<MemRefType>().getShape();
    int64_t dataRank = dataShape.size();
    Value indices = operandAdaptor.indices();
    auto indicesShape = indices.getType().cast<MemRefType>().getShape();
    int64_t indicesRank = indicesShape.size();
    int64_t axisIndex = gatherOp.axis().getSExtValue();
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
    printf("hi alex, got info axis %d and with sized data %d, indices %d, and "
           "output %d\n",
        (int)axisIndex, (int)dataRank, (int)indicesRank, (int)outputRank);
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, outputRank);
    printf("hi alex, created loop vars\n");
    KrnlIterateOperandPack pack(rewriter, originalLoops);
    int iIndexStart = 0;
    for (int ii = 0; ii < axisIndex; ++ii)
      addDimensionToPack(rewriter, loc, pack, data, ii);
    // Then iterates over the Nj (indices matrix), jj indices in above algo.
    int jIndexStart = iIndexStart + axisIndex;
    for (int jj = 0; jj < indicesRank; ++jj)
      addDimensionToPack(rewriter, loc, pack, indices, jj);
    // Finally iterates over the Nk (data after axis), kk indices in above algo.
    int kIndexStart = jIndexStart + indicesRank - (axisIndex+1);
    for (int kk = axisIndex + 1; kk < dataRank; ++kk)
      addDimensionToPack(rewriter, loc, pack, data, kk);
    printf("hi alex, created %d loops\n", (int)outputRank);
    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(outputMemRefType))
      alloc =
          insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);
    else
      return emitError(loc, "unsupported dynamic dimensions");
    printf("hi alex, created alloc\n");

    // Create the loops
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();
    printf("hi alex, created iterate\n");

    // Now perform the insertions into the body of the just generated loops.
    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operations.
    // Read first the indices[jj] into indexVal.
    SmallVector<Value, 4> indicesMemRefVal;
    for (int j = 0; j < indicesRank; ++j)
      indicesMemRefVal.emplace_back(
          iterationBlock.getArguments()[jIndexStart + j]);
    auto indexValInteger =
        rewriter.create<AffineLoadOp>(loc, indices, indicesMemRefVal);
    printf("hi alex, created load indices\n");
    auto indexVal = rewriter.create<IndexCastOp>(loc, indexValInteger, rewriter.getIndexType());
    printf("hi alex, created casted load indices\n");

    // Then read input data into DataVal: first add ii's.
    SmallVector<Value, 4> dataMemRefVal;
    for (int i = 0; i < axisIndex; ++i)
      dataMemRefVal.emplace_back(
          iterationBlock.getArguments()[iIndexStart + i]);
    // Then add indices[jj] (indexVal)
    dataMemRefVal.emplace_back(indexVal);
    // Then add kk's
    for (int k = axisIndex + 1; k < dataRank; ++k) {
      printf("  %d: emplace loop index %d\n", k, kIndexStart +k);
      dataMemRefVal.emplace_back(
          iterationBlock.getArguments()[kIndexStart + k]);
    }
    auto dataVal = rewriter.create<AffineLoadOp>(loc, data, dataMemRefVal);
    printf("hi alex, created load data\n");

    // Then store the value in the output.
    SmallVector<Value, 4> outputMemRefVal;
    for (int n = 0; n < iterationBlock.getArguments().size(); ++n)
      outputMemRefVal.emplace_back(iterationBlock.getArguments()[n]);

    rewriter.create<AffineStoreOp>(loc, dataVal, alloc, outputMemRefVal);
    printf("hi alex, created store output\n");

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXGatherOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXGatherOpLowering>(ctx);
}
