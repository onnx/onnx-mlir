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

using namespace mlir;

struct ONNXTileOpLowering : public ConversionPattern {
  ONNXTileOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTileOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTileOpAdaptor operandAdaptor(operands);
    ONNXTileOp gatherOp = llvm::cast<ONNXTileOp>(op);
    auto loc = op->getLoc();
    // get input operands, shapes, and rank
    Value input = operandAdaptor.input();
    auto inputShape = input.getType().cast<MemRefType>().getShape();
    int64_t inputRank = inputShape.size();

    Value repeats = operandAdaptor.repeats();
    auto repeatsMemRefType = repeats.getType().cast<MemRefType>();
    auto repeatsShape = repeats.getType().cast<MemRefType>().getShape();
    int64_t repeatsRank = repeatsShape.size();

    // get output info
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputMemRefShape = outputMemRefType.getShape();
    int64_t outputRank = outputMemRefShape.size();

    bool insertDealloc = checkInsertDealloc(op);
    Value alloc;
    if (hasAllConstantDimensions(outputMemRefType))
      alloc =
          insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);
    else
      return emitError(loc, "unsupported dynamic dimensions");

    // Define loops and iteration trip counts (equivalent to size of output)
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, outputRank);
    KrnlIterateOperandPack pack(rewriter, originalLoops);
    for (int ii = 0; ii < outputRank; ++ii)
      addDimensionToPack(rewriter, loc, pack, alloc, ii);

    // Create the loops
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the just generated loops.
    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operations.

    // This implementation is to iterate the output tensor.
    // The store has simple affine subscript expression.
    // Alternative implementation is to iterate the input tensor and repeats.
    // The load of elements in input tensor can be reused explicitly.
    // But the subscript of store is not contigous, or even not affine.
    SmallVector<Value, 4> inputMemRefVal;
    for (int j = 0; j < outputRank; ++j) {
      auto indexVal = emitConstantOp(rewriter, loc, rewriter.getIndexType(), j);
      SmallVector<Value, 1> repeatsMemRefVal = {indexVal};
      auto repeatsElementVal =
          rewriter.create<AffineLoadOp>(loc, repeats, repeatsMemRefVal);
      auto repeatsElementConvertedVal = rewriter.create<IndexCastOp>(
          loc, repeatsElementVal, rewriter.getIndexType());
      auto loopVarVal = iterationBlock.getArguments()[j];
      auto exprVal = rewriter.create<UnsignedRemIOp>(
          loc, loopVarVal, repeatsElementConvertedVal);
      inputMemRefVal.emplace_back(exprVal);
    }
    auto inputVal = rewriter.create<AffineLoadOp>(loc, input, inputMemRefVal);

    SmallVector<Value, 4> outputMemRefVal(iterationBlock.getArguments().begin(),
        iterationBlock.getArguments().end());

    // Then store the value in the output.
    rewriter.create<AffineStoreOp>(loc, inputVal, alloc, outputMemRefVal);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXTileOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTileOpLowering>(ctx);
}
