//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Concat Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXConcatOpLowering : public ConversionPattern {
  ONNXConcatOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConcatOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    auto loc = op->getLoc();
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    ONNXConcatOp concatOp = llvm::dyn_cast<ONNXConcatOp>(op);
    auto axis = concatOp.axis().getSExtValue();
    int inputNum = operands.size();
    // Alloc and dealloc.
    auto resultOperand = concatOp.concat_result();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto resultShape = memRefType.getShape();
    auto rank = resultShape.size();
    assert((axis >=0 && axis < rank) && "Concat axis out of bounds");

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, {resultOperand});

    // Creates loops, one for each input.
    int writeOffset = 0;
    for (int i = 0; i < inputNum; ++i) {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      // Operand info.
      auto currShape = operands[i].getType().cast<MemRefType>().getShape();
      // Create loop.
      BuildKrnlLoop inputLoops(rewriter, loc, rank);
      inputLoops.createDefineAndOptimizeOp();
      for (int r = 0; r < rank; ++r)
        inputLoops.pushBounds(0, operands[i], r);
      inputLoops.createIterateOp();
      rewriter.setInsertionPointToStart(inputLoops.getIterateBlock());
      // Indices for the read and write.
      SmallVector<Value, 4> readIndices;
      SmallVector<Value, 4> writeIndices;
      for (int r = 0; r < rank; ++r) {
        readIndices.emplace_back(inputLoops.getInductionVar(r));
        if (r != axis || writeOffset == 0) {
          writeIndices.emplace_back(inputLoops.getInductionVar(r));
        } else {
          auto indexWithOffset = rewriter.create<AddIOp>(loc,
              rewriter.create<ConstantIndexOp>(loc, writeOffset),
              inputLoops.getInductionVar(r));
          writeIndices.emplace_back(indexWithOffset);
        }
      }
      // Insert copy.
      auto loadData = rewriter.create<LoadOp>(loc, operands[i], readIndices);
      rewriter.create<StoreOp>(loc, loadData, alloc, writeIndices);
      // Increment offset
      writeOffset += currShape[axis];
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXConcatOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConcatOpLowering>(ctx);
}
