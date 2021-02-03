//===----------------Tile.cpp - Lowering Tile Op----------------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Tile Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper function to insert alloc and dealloc ops for memref of dynamic shape.
//

Value insertAllocAndDeallocForTile(MemRefType memRefType, Location loc,
    ConversionPatternRewriter &rewriter, bool insertDealloc, Value inputOperand,
    Value repeatsOperand) {
  AllocOp alloc;
  auto inputShape = inputOperand.getType().cast<MemRefType>().getShape();
  auto inputRank = inputShape.size();
  auto outputShape = memRefType.getShape();

  SmallVector<Value, 4> allocOperands;
  for (int i = 0; i < inputRank; ++i) {
    if (outputShape[i] == -1) {
      auto indexVal = emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
      SmallVector<Value, 1> repeatsMemRefVal = {indexVal};
      auto repeatsLoadVal =
          rewriter.create<KrnlLoadOp>(loc, repeatsOperand, repeatsMemRefVal);
      auto repeatsElementVal = rewriter.create<IndexCastOp>(
          loc, repeatsLoadVal, rewriter.getIndexType());
      auto dimVal = rewriter.create<DimOp>(loc, inputOperand, i);
      Value allocDimVal =
          rewriter.create<MulIOp>(loc, dimVal, repeatsElementVal);
      allocOperands.emplace_back(allocDimVal);
    }
  }
  alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
  if (insertDealloc) {
    auto *parentBlock = alloc.getOperation()->getBlock();
    auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }
  return alloc;
}

struct ONNXTileOpLowering : public ConversionPattern {
  ONNXTileOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTileOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTileOpAdaptor operandAdaptor(operands);
    ONNXTileOp tileOp = llvm::cast<ONNXTileOp>(op);
    auto loc = op->getLoc();

    ONNXTileOpShapeHelper shapeHelper(&tileOp, &rewriter);

    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    (void)shapecomputed;
    assert(!failed(shapecomputed) && "expected to succeed");

    auto resultOperand = tileOp.output();
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputMemRefShape = outputMemRefType.getShape();
    int64_t outputRank = outputMemRefShape.size();

    Value input = operandAdaptor.input();

    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Define loops and iteration trip counts (equivalent to size of output)
    BuildKrnlLoop outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    SmallVector<Value, 4> loadIndices;
    bool isAffineLoad = true;

    // This implementation is to iterate the output tensor.
    // The store has simple affine subscript expression.
    // Alternative implementation is to iterate the input tensor and repeats.
    // The load of elements in input tensor can be reused explicitly.
    // But the subscript of store is not contigous, or even not affine.
    // Alternative implementation can be found at the end of this file.

    for (int i = 0; i < outputRank; i++) {
      // Context is created for each dimension because they are independent
      IndexExprContext IEContext(&rewriter, loc);
      Value loopVal = outputLoops.getInductionVar(i);
      IndexExpr index = IEContext.createLoopInductionIndex(loopVal);
      IndexExpr dimSize = IEContext.createDimIndexFromShapedType(input, i);
      IndexExpr exprVal = index % dimSize;
      if (!exprVal.isAffine()) {
        isAffineLoad = false;
      }
      loadIndices.emplace_back(exprVal.getValue());
    }

    Value loadVal = rewriter.create<KrnlLoadOp>(loc, input, loadIndices);

    SmallVector<Value, 4> storeIndices;
    for (int i = 0; i < outputRank; ++i) {
      storeIndices.emplace_back(outputLoops.getInductionVar(i));
    }
    rewriter.create<KrnlStoreOp>(loc, loadVal, alloc, storeIndices);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

// This is the alternative way of lowering.
// It is kept here for record in case this implementation is needed
struct ONNXTileOpLoweringAlternative : public ConversionPattern {
  ONNXTileOpLoweringAlternative(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTileOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTileOpAdaptor operandAdaptor(operands);
    ONNXTileOp tileOp = llvm::cast<ONNXTileOp>(op);
    auto loc = op->getLoc();
    // get input operands, shapes, and rank
    Value input = operandAdaptor.input();
    auto inputShape = input.getType().cast<MemRefType>().getShape();
    int64_t inputRank = inputShape.size();
    Value repeats = operandAdaptor.repeats();

    // get output info
    auto resultOperand = tileOp.output();
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputMemRefShape = outputMemRefType.getShape();
    int64_t outputRank = outputMemRefShape.size();

    bool insertDealloc = checkInsertDealloc(op);
    Value alloc;
    if (hasAllConstantDimensions(outputMemRefType))
      alloc =
          insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDeallocForTile(
          outputMemRefType, loc, rewriter, insertDealloc, input, repeats);

    // Define loops and iteration trip counts (equivalent to size of output)
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, outputRank * 2);
    KrnlIterateOperandPack pack(rewriter, originalLoops);
    for (int ii = 0; ii < outputRank; ++ii) {
      addDimensionToPack(rewriter, loc, pack, input, ii);
      pack.pushConstantBound(0);
      auto indexVal =
          emitConstantOp(rewriter, loc, rewriter.getIndexType(), ii);
      SmallVector<Value, 1> repeatsMemRefVal = {indexVal};
      auto repeatsLoadVal =
          rewriter.create<KrnlLoadOp>(loc, repeats, repeatsMemRefVal);
      auto repeatsElementVal = rewriter.create<IndexCastOp>(
          loc, repeatsLoadVal, rewriter.getIndexType());
      pack.pushOperandBound(repeatsElementVal);
    }

    // Create the loops
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the just generated loops.
    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operations.

    SmallVector<Value, 4> inputMemRefVal;
    for (int j = 0; j < inputRank; ++j) {
      inputMemRefVal.emplace_back(iterationBlock.getArguments()[j * 2]);
    }

    SmallVector<Value, 4> outputMemRefVal;
    for (int i = 0; i < inputRank; ++i) {
      auto inputDimSizeVal = rewriter.create<DimOp>(loc, input, i);
      if (inputShape[i] != -1) {
        auto inputIndexAE = rewriter.getAffineDimExpr(0);
        auto repeatsIndexAE = rewriter.getAffineDimExpr(1);
        auto inputDimAE = rewriter.getAffineSymbolExpr(0);

        auto dimMap =
            AffineMap::get(2, 1, inputDimAE * repeatsIndexAE + inputIndexAE);
        auto dimExprVal = rewriter.create<AffineApplyOp>(loc, dimMap,
            ArrayRef<Value>{iterationBlock.getArguments()[2 * i],
                iterationBlock.getArguments()[2 * i + 1], inputDimSizeVal});
        outputMemRefVal.emplace_back(dimExprVal);
      } else {
        auto inputIndex = iterationBlock.getArguments()[2 * i];
        auto repeatsIndex = iterationBlock.getArguments()[2 * i + 1];
        auto dimExprVal = rewriter.create<AddIOp>(loc, inputIndex,
            rewriter.create<MulIOp>(loc, repeatsIndex, inputDimSizeVal));
        outputMemRefVal.emplace_back(dimExprVal);
      }
    }

    auto inputVal = rewriter.create<KrnlLoadOp>(loc, input, inputMemRefVal);
    rewriter.create<KrnlStoreOp>(loc, inputVal, alloc, outputMemRefVal);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXTileOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTileOpLowering>(ctx);
}
