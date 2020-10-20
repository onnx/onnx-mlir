//===---------------- Split.cpp - Lowering Split Op -----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Split Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXSplitOpLowering : public ConversionPattern {
  ONNXSplitOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSplitOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    auto loc = op->getLoc();
    ONNXSplitOp splitOp = llvm::dyn_cast<ONNXSplitOp>(op);
    auto axis = splitOp.axis();
    auto split = splitOp.split().getValue();
    SmallVector<int64_t, 4> splitOffset;
    int64_t offset = 0;
    for (int i = 0; i < split.size(); ++i) {
      splitOffset.emplace_back(offset);
      offset += ArrayAttrIntVal(split, i);
    }
    auto rank = splitOp.input().getType().cast<ShapedType>().getRank();
    auto outputNum = splitOp.getNumResults();

    // Alloc and dealloc.
    SmallVector<Value, 4> allocs;
    for (int i = 0; i < outputNum; ++i) {
      Value alloc;
      bool insertDealloc = checkInsertDealloc(op, i);
      auto memRefType = convertToMemRefType(splitOp.outputs()[i].getType());

      if (hasAllConstantDimensions(memRefType))
        alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
      else {
        SmallVector<Value, 4> allocOperands;
        auto shape = memRefType.getShape();
        for (decltype(rank) r = 0; r < rank; ++r) {
          if (shape[r] < 0) {
            Value dim;
            if (r != axis)
              dim = rewriter.create<DimOp>(loc, operands[0], r);
            else
              dim = emitConstantOp(rewriter, loc, rewriter.getIndexType(),
                  ArrayAttrIntVal(split, i));
            allocOperands.push_back(dim);
          }
        }
        alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
        if (insertDealloc) {
          auto *parentBlock = alloc.getDefiningOp()->getBlock();
          auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
          dealloc.getOperation()->moveBefore(&parentBlock->back());
        }
      }
      allocs.emplace_back(alloc);
    }

    // Creates loops, one for each output.
    for (int i = 0; i < outputNum; ++i) {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Create loop.
      BuildKrnlLoop outputLoops(rewriter, loc, rank);
      outputLoops.createDefineAndIterateOp(allocs[i]);
      rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());
      // Indices for the read and write.
      SmallVector<Value, 4> readIndices;
      SmallVector<Value, 4> writeIndices;
      for (int r = 0; r < rank; ++r) {
        // Same index for read and write if the dimension is:
        //  - the first dimension, or
        //  - not the split axis.
        if (i == 0 || r != axis) {
          readIndices.emplace_back(outputLoops.getInductionVar(r));
        } else {
          auto index = rewriter.getAffineDimExpr(0);
          auto indexMap = AffineMap::get(1, 0, index + splitOffset[i]);
          auto indexWithOffset = rewriter.create<AffineApplyOp>(loc, indexMap,
              ArrayRef<Value>{/*index=*/outputLoops.getInductionVar(r)});
          readIndices.emplace_back(indexWithOffset);
        }
        writeIndices.emplace_back(outputLoops.getInductionVar(r));
      }
      // Insert copy.
      auto loadData =
          rewriter.create<AffineLoadOp>(loc, operands[0], readIndices);
      rewriter.create<AffineStoreOp>(loc, loadData, allocs[i], writeIndices);
    }
    rewriter.replaceOp(op, allocs);
    return success();
  }
};

void populateLoweringONNXSplitOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLowering>(ctx);
}
