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
    auto axis = splitOp.axis().getSExtValue();
    SmallVector<int64_t, 4> splitOffset;
    int64_t offset = 0;
    for (int i = 0; i < splitOp.split().getValue().size(); ++i) {
      splitOffset.emplace_back(offset);
      offset += ArrayAttrIntVal(splitOp.split().getValue(), i);
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
      else
        // TODO: unknown dimensions.
        return failure();
      // alloc = insertAllocAndDealloc(
      //    memRefType, loc, rewriter, insertDealloc, {splitOp.outputs()[i]});
      allocs.emplace_back(alloc);
    }

    // Creates loops, one for each output.
    for (int i = 0; i < outputNum; ++i) {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Create loop.
      BuildKrnlLoop outputLoops(rewriter, loc, rank);
      outputLoops.createDefineAndOptimizeOp();
      for (int r = 0; r < rank; ++r)
        outputLoops.pushBounds(0, allocs[i], r);
      outputLoops.createIterateOp();
      rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());
      // Indices for the read and write.
      SmallVector<Value, 4> readIndices;
      SmallVector<Value, 4> writeIndices;
      for (int r = 0; r < rank; ++r) {
        if (r != axis || i == 0) {
          readIndices.emplace_back(outputLoops.getInductionVar(r));
        } else {
          AffineMap indexWithOffsetMap = AffineMap::get(
              1, 0, rewriter.getAffineDimExpr(0) + splitOffset[i]);
          auto indexWithOffset =
              rewriter.create<AffineApplyOp>(loc, indexWithOffsetMap,
                  ValueRange(ArrayRef<Value>{outputLoops.getInductionVar(r)}));
          readIndices.emplace_back(indexWithOffset);
        }
        writeIndices.emplace_back(outputLoops.getInductionVar(r));
      }
      // Insert copy.
      auto loadData = rewriter.create<LoadOp>(loc, operands[0], readIndices);
      rewriter.create<StoreOp>(loc, loadData, allocs[i], writeIndices);
    }
    rewriter.replaceOp(op, allocs);
    return success();
  }
};

void populateLoweringONNXSplitOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLowering>(ctx);
}
