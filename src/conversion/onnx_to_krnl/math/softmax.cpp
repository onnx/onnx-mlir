//===----- softmax.cpp - Softmax Op ---------------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/conversion/onnx_to_krnl/onnx_to_krnl_common.hpp"

using namespace mlir;

struct ONNXSoftmaxOpLowering : public ConversionPattern {
  ONNXSoftmaxOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSoftmaxOp::getOperationName(), 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // softmax(x) = let max_x = max(x) in
    //                let exp_x = exp(x - max_x) in
    //                  let sum = sum(exp_x) in
    //                    exp_x / sum
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int64_t rank = memRefType.getRank();
    int64_t axis = llvm::dyn_cast<ONNXSoftmaxOp>(op).axis().getSExtValue();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);

    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto elementType = memRefType.getElementType();

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc,
                                    operands[0]);

    // Shape of the result
    auto memRefShape = memRefType.getShape();

    // Insert allocations and deallocations for sum and max.
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    Value sumOp = insertAllocAndDealloc(scalarMemRefType, loc, rewriter, true);
    Value maxOp = insertAllocAndDealloc(scalarMemRefType, loc, rewriter, true);
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    Value negInfinity = rewriter.create<ConstantOp>(
        loc,
        FloatAttr::get(elementType, -std::numeric_limits<float>::infinity()));

    // Define loops.
    std::vector<Value> originalLoops;
    std::vector<Value> optimizedLoops;
    Block *optimizationBlock = defineLoops(rewriter, loc, originalLoops,
            optimizedLoops, rank);

    // Coerce the input into a 2-D tensor. `axis` will be the coercing point.
    // This coercing follows the softmax definition in ONNX:
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
    // Here, we create an outer loop and inner loop for handling the two
    // dimensions. The outer loop is only created once `axis` is not zero.

    // Define an outer loop with respect to axis.
    std::vector<Value> outerLoops, optimizedOuterLoops;
    outerLoops.reserve(axis);
    optimizedOuterLoops.reserve(axis);
    for (int i = 0; i < axis; ++i) {
      outerLoops.push_back(originalLoops[i]);
      optimizedOuterLoops.push_back(optimizedLoops[i]);
    }
    KrnlIterateOperandPack outerPack(rewriter, outerLoops, optimizedOuterLoops);
    for (int i = 0; i < axis; ++i)
      addDimensionToPack(rewriter, loc, outerPack, operands[0], i);

    // Define an inner loop with respect to axis.
    std::vector<Value> innerLoops, optimizedInnerLoops;
    innerLoops.reserve(rank - axis);
    optimizedInnerLoops.reserve(rank - axis);
    for (int i = axis; i < rank; ++i) {
      innerLoops.push_back(originalLoops[i]);
      optimizedInnerLoops.push_back(optimizedLoops[i]);
    }
    KrnlIterateOperandPack innerPack(rewriter, innerLoops, optimizedInnerLoops);
    for (int i = axis; i < rank; ++i)
      addDimensionToPack(rewriter, loc, innerPack, operands[0], i);

    KrnlIterateOp outerIterateOp, maxIterateOp, sumIterateOp, softmaxIterateOp;
    SmallVector<Value, 4> outerLoopIVs;
    if (axis != 0) {
      outerIterateOp = rewriter.create<KrnlIterateOp>(loc, outerPack);

      // No optimization
      rewriter.setInsertionPointToEnd(optimizationBlock);
      rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);

      // Insert instructions inside the outer loop.
      Block &outerIterationBlock = outerIterateOp.bodyRegion().front();
      rewriter.setInsertionPointToStart(&outerIterationBlock);
      for (auto arg : outerIterationBlock.getArguments())
        outerLoopIVs.push_back(arg);

      // Reset accumulators.
      rewriter.create<StoreOp>(loc, zero, sumOp);
      rewriter.create<StoreOp>(loc, negInfinity, maxOp);

      // Create an inner loop to compute max.
      maxIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);
      // Create an inner loop to compute sum.
      sumIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);
      // Create an inner loop to compute softmax.
      softmaxIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);
    } else {
      // Reset accumulators.
      rewriter.create<StoreOp>(loc, zero, sumOp);
      rewriter.create<StoreOp>(loc, negInfinity, maxOp);

      // Create an inner loop to compute max.
      maxIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);
      // Create an inner loop to compute sum.
      sumIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);
      // Create an inner loop to compute softmax.
      softmaxIterateOp = rewriter.create<KrnlIterateOp>(loc, innerPack);

      // No optimization
      rewriter.setInsertionPointToEnd(optimizationBlock);
      rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);
    }

    // Insert instructions inside the max loop.
    Block &maxIterationBlock = maxIterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&maxIterationBlock);

    // Get induction variables.
    SmallVector<Value, 4> maxLoopIVs;
    for (auto arg : outerLoopIVs)
      maxLoopIVs.push_back(arg);
    for (auto arg : maxIterationBlock.getArguments())
      maxLoopIVs.push_back(arg);

    // Compute the max value.
    Value max = rewriter.create<LoadOp>(loc, maxOp);
    Value nextMax = rewriter.create<LoadOp>(loc, operands[0], maxLoopIVs);
    auto maxCond =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, max, nextMax);
    max = rewriter.create<SelectOp>(loc, maxCond, max, nextMax);
    rewriter.create<StoreOp>(loc, max, maxOp);

    // Get the max.
    rewriter.setInsertionPoint(sumIterateOp);
    max = rewriter.create<LoadOp>(loc, maxOp);

    // Insert instructions inside the sum loop.
    Block &sumIterationBlock = sumIterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&sumIterationBlock);

    // Get induction variables.
    SmallVector<Value, 4> sumLoopIVs;
    for (auto arg : outerLoopIVs)
      sumLoopIVs.push_back(arg);
    for (auto arg : sumIterationBlock.getArguments())
      sumLoopIVs.push_back(arg);

    // Sum up values.
    Value sum = rewriter.create<LoadOp>(loc, sumOp);
    Value next = rewriter.create<LoadOp>(loc, operands[0], sumLoopIVs);
    Value sub = rewriter.create<SubFOp>(loc, next, max);
    Value exp = rewriter.create<ExpOp>(loc, sub);
    sum = rewriter.create<AddFOp>(loc, sum, exp);
    rewriter.create<StoreOp>(loc, sum, sumOp);
    // Store intermediate values in the result to avoid recomputation.
    rewriter.create<StoreOp>(loc, exp, alloc, sumLoopIVs);

    // Get the sum.
    rewriter.setInsertionPoint(softmaxIterateOp);
    sum = rewriter.create<LoadOp>(loc, sumOp);

    // Insert instructions inside the softmax loop.
    Block &softmaxIterationBlock = softmaxIterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&softmaxIterationBlock);

    // Get induction variables.
    SmallVector<Value, 4> softmaxLoopIVs;
    for (auto arg : outerLoopIVs)
      softmaxLoopIVs.push_back(arg);
    for (auto arg : softmaxIterationBlock.getArguments())
      softmaxLoopIVs.push_back(arg);

    // Compute softmax.
    Value expLoadedVal = rewriter.create<LoadOp>(loc, alloc, softmaxLoopIVs);
    Value result = rewriter.create<DivFOp>(loc, expLoadedVal, sum);
    rewriter.create<StoreOp>(loc, result, alloc, softmaxLoopIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

void populateLoweringONNXSoftmaxOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSoftmaxOpLowering>(ctx);
}
