//===---------------- Flatten.cpp - Lowering Flatten Op
//-------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Flatten Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper function to insert alloc and dealloc ops for memref of dynamic shape.
//
// Should namespace or static be used here?
Value insertAllocAndDeallocForFlatten(MemRefType memRefType, Location loc,
    ConversionPatternRewriter &rewriter, bool insertDealloc, Value input,
    int64_t axisValue) {
  AllocOp alloc;
  auto inputShape = input.getType().cast<MemRefType>().getShape();
  auto inputRank = inputShape.size();

  SmallVector<Value, 2> allocOperands;
  auto firstDimVal = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);
  auto secondDimVal = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);
  for (auto i = 0; i < inputRank; i++) {
    if (i < axisValue) 
      firstDimVal = rewriter.create<MulIOp>(loc, firstDimVal,
          rewriter.create<DimOp>(loc, input, i));
    else 
      secondDimVal = rewriter.create<MulIOp>(loc, secondDimVal,
          rewriter.create<DimOp>(loc, input, i));
  }
  allocOperands.emplace_back(firstDimVal);
  allocOperands.emplace_back(secondDimVal);

  alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
  if (insertDealloc) {
    auto *parentBlock = alloc.getOperation()->getBlock();
    auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }
  return alloc;
}

struct ONNXFlattenOpLowering : public ConversionPattern {
  ONNXFlattenOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXFlattenOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Gather info.
    Location loc = op->getLoc();
    ONNXFlattenOp flattenOp = llvm::dyn_cast<ONNXFlattenOp>(op);

    ONNXFlattenOpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.input();
    auto inputTy = input.getType().cast<MemRefType>();
    auto inputShape = inputTy.getShape();
    auto inputRank = inputShape.size();
    auto axisValue = flattenOp.axis();
    if (axisValue < 0)
      axisValue = inputRank + axisValue + 1;

    // Insert alloc and dealloc
    bool insertDealloc = checkInsertDealloc(op);
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc;
    if (hasAllConstantDimensions(outputMemRefType))
      alloc =
          insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDeallocForFlatten(
          outputMemRefType, loc, rewriter, insertDealloc, input, axisValue);

    // Define loops and iteration trip counts (equivalent to size of input)
    ValueRange indices;
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, inputRank);
    KrnlIterateOperandPack pack(rewriter, originalLoops);
    for (int i = 0; i < inputRank; ++i)
      addDimensionToPack(rewriter, loc, pack, input, i);

    // Create the loops
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the just generated loops.
    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Generate the load of input
    SmallVector<Value, 4> inputMemRefVal(iterationBlock.getArguments().begin(),
        iterationBlock.getArguments().end());
    auto inputVal = rewriter.create<AffineLoadOp>(loc, input, inputMemRefVal);
   
    // Generate the store for output
    // Define affine map for first dim of output
    SmallVector<Value, 4> firstMapArgList;
    auto firstIndexAE = rewriter.getAffineConstantExpr(0);
    auto firstAccumulatedDimSizeAE = rewriter.getAffineConstantExpr(1);
    for (auto i = axisValue-1 ; i >= 0 ; i--) {
      auto dimIndexAE = rewriter.getAffineDimExpr(i);
      firstIndexAE = firstIndexAE + dimIndexAE * firstAccumulatedDimSizeAE;
      auto dimSizeAE = rewriter.getAffineSymbolExpr(i);
      firstAccumulatedDimSizeAE = dimSizeAE * firstAccumulatedDimSizeAE;
    }
    AffineMap firstDimMap = AffineMap::get(axisValue, axisValue, firstIndexAE);

    // Create the parameter lists for the affine map
    for (auto i = 0 ; i < axisValue; i++) {
      firstMapArgList.emplace_back(iterationBlock.getArguments()[i]);
    }
    for (auto i = 0 ; i < axisValue; i++) {
      firstMapArgList.emplace_back(rewriter.create<DimOp>(loc, input, i));
    }
    auto firstDimVal = rewriter.create<AffineApplyOp>(
          loc, firstDimMap, firstMapArgList);

    // Generate index for second dim of output
    SmallVector<Value, 4> secondMapArgList;
    auto secondIndexAE = rewriter.getAffineConstantExpr(0);
    auto secondAccumulatedDimSizeAE = rewriter.getAffineConstantExpr(1);
    for (auto i = inputRank-1 ; i >= axisValue ; i--) {
      auto idx = i - axisValue;
      auto dimIndexAE = rewriter.getAffineDimExpr(idx);
      secondIndexAE = secondIndexAE + dimIndexAE * secondAccumulatedDimSizeAE;
      auto dimSizeAE = rewriter.getAffineSymbolExpr(idx);
      secondAccumulatedDimSizeAE = dimSizeAE * secondAccumulatedDimSizeAE;
    }
    AffineMap secondDimMap = AffineMap::get(inputRank-axisValue, inputRank-axisValue, secondIndexAE);

    // Create the parameter lists for the affine map
    for (auto i = axisValue ; i < inputRank; i++) {
      secondMapArgList.emplace_back(iterationBlock.getArguments()[i]);
    }
    for (auto i = axisValue ; i < inputRank; i++) {
      secondMapArgList.emplace_back(rewriter.create<DimOp>(loc, input, i));
    }
    auto secondDimVal = rewriter.create<AffineApplyOp>(
          loc, secondDimMap, secondMapArgList);

    // Create the store 
    SmallVector<Value, 2> outputMemRefVal = {firstDimVal, secondDimVal};
    rewriter.create<AffineStoreOp>(loc, inputVal, alloc, outputMemRefVal);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXFlattenOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXFlattenOpLowering>(ctx);
}
