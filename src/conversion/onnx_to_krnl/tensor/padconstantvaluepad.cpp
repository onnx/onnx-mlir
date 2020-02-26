//===----- transpose.inc - Lowering Transpose Op --------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Transpose Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/conversion/onnx_to_krnl/onnx_to_krnl_common.hpp"

using namespace mlir;

struct ONNXPadConstantValuePadOpLowering : public ConversionPattern {
  ONNXPadConstantValuePadOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXPadConstantValuePadOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    //Only constant padding is supported now.
    auto padMode = llvm::dyn_cast<ONNXPadConstantValuePadOp>(op).mode();
    if (padMode != "constant")
      emitError(loc, "unsupported mode for PadConstantValuePad");
    auto constantValAttr = llvm::dyn_cast<ONNXPadConstantValuePadOp>(op).constant_valueAttr();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(tensorType);
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      emitError(loc, "unexpected output has non-Constant shape");

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t rank = memRefShape.size();

    // Copy the input data into the output.

    // Define loops.
    std::vector<Value> originalLoops;
    std::vector<Value> optimizedLoops;
    KrnlOptimizeLoopsOp optimizedLoopsOp = emitOptimizedLoops(rewriter, loc, originalLoops,
        optimizedLoops, rank);
    Block *optimizationBlock = &optimizedLoopsOp.region().front();

    // Iterate over the loop nest using the input shape.
    KrnlIterateOperandPack pack(rewriter, originalLoops, optimizedLoops);
    for (int i = 0; i < rank; ++i)
      addDimensionToPack(rewriter, loc, pack, operands[0], i);

    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the
    // just generated instructions:

    // 1. Insert any optimizations in the KrnlOptimizeLoopsOp body.
    rewriter.setInsertionPointToEnd(optimizationBlock);
    // Return from KrnlOptimizeLoopsOp body.
    // When no optimizations are present we just return the loops
    // unchaged.
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);

    // 2. Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation.
    SmallVector<Value, 4> inLoopIVs;
    for (auto arg : iterationBlock.getArguments())
      inLoopIVs.emplace_back(arg);

    auto pads = llvm::dyn_cast<ONNXPadConstantValuePadOp>(op).pads();
    SmallVector<int64_t, 4> pad_begin;
    for (int i=0; i<pads.size(); i+=2 ) {
        pad_begin.emplace_back(pads.getValue()[i].cast<IntegerAttr>().getInt());
    }

    SmallVector<Value, 4> outLoopIVs;
    for (int i=0; i<iterationBlock.getArguments().size(); ++i) {
      //Calculate the index.
      if (pad_begin[i] == 0) {
        outLoopIVs.emplace_back(iterationBlock.getArguments()[i]);
      }else {
        auto outIV = rewriter.create<AddIOp>(loc, rewriter.create<ConstantIndexOp>(loc, pad_begin[i]), iterationBlock.getArguments()[i]);
        outLoopIVs.emplace_back(outIV);
      }
    }

    auto inVal = rewriter.create<LoadOp>(loc, operands[0], inLoopIVs);
    rewriter.create<StoreOp>(loc, inVal, alloc, outLoopIVs);
    rewriter.setInsertionPoint(optimizedLoopsOp);

    //Copy padding value into the output

    // Define loops.
    std::vector<Value> originalLoops1;
    std::vector<Value> optimizedLoops1;
    Block *optimizationBlock1 = defineLoops(rewriter, loc, originalLoops1,
        optimizedLoops1, rank);
    KrnlIterateOperandPack pack1(rewriter, originalLoops1, optimizedLoops1);

    // Iterate over the loop nest using the input shape.
    for (int i = 0; i < rank; ++i) {
      pack1.pushConstantBound(0);
      pack1.pushConstantBound(memRefShape[i]);
    }

    auto iterateOp1 = rewriter.create<KrnlIterateOp>(loc, pack1);
    Block &iterationBlock1 = iterateOp1.bodyRegion().front();

    rewriter.setInsertionPointToEnd(optimizationBlock1);
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops1);
    rewriter.setInsertionPointToStart(&iterationBlock1);
    SmallVector<Value, 4> outLoopIVs1;
    for (auto arg : iterationBlock1.getArguments())
      outLoopIVs1.emplace_back(arg);

    auto inVal1 = rewriter.create<ConstantOp>(loc, constantValAttr);
    rewriter.create<StoreOp>(loc, inVal1, alloc, outLoopIVs1);


    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

void populateLoweringONNXPadConstantValuePadOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPadConstantValuePadOpLowering>(ctx);
}
