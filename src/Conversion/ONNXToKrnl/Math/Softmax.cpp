/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Softmax.cpp - Softmax Op ---------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

struct ONNXSoftmaxOpLowering : public ConversionPattern {
  ONNXSoftmaxOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSoftmaxOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // softmax(x) = let max_x = max(x) in
    //                let exp_x = exp(x - max_x) in
    //                  let sum = sum(exp_x) in
    //                    exp_x / sum
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int64_t rank = memRefType.getRank();
    int64_t axis = llvm::dyn_cast<ONNXSoftmaxOp>(op).axis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);

    auto loc = op->getLoc();
    ONNXSoftmaxOpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.input();
    // Insert an allocation and deallocation for the result of this operation.
    auto elementType = memRefType.getElementType();

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, input);

    // Shape of the result
    auto memRefShape = memRefType.getShape();

    // Insert allocations and deallocations for sum and max.
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    Value sumOp = insertAllocAndDealloc(scalarMemRefType, loc, rewriter, true);
    Value maxOp = insertAllocAndDealloc(scalarMemRefType, loc, rewriter, true);
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    Value negInfinity = rewriter.create<ConstantOp>(loc,
        FloatAttr::get(elementType, -std::numeric_limits<float>::infinity()));

    // Define loops.
    std::vector<Value> originalLoops;

    // Coerce the input into a 2-D tensor. `axis` will be the coercing point.
    // This coercing follows the softmax definition in ONNX:
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
    // Here, we create an outer loop and inner loop for handling the two
    // dimensions. The outer loop is only created once `axis` is not zero.

    BuildKrnlLoop buildOuterLoops(rewriter, loc, /*loopNum=*/axis);
    buildOuterLoops.createDefineOp();
    for (int i = 0; i < axis; ++i)
      buildOuterLoops.pushBounds(0, input, i);
    buildOuterLoops.createIterateOp();
    std::vector<Value> outerLoops = buildOuterLoops.getOriginalLoops();

    BuildKrnlLoop buildInnerMaxLoop(rewriter, loc, /*loopNum=*/rank - axis);
    BuildKrnlLoop buildInnerSumLoop(rewriter, loc, /*loopNum=*/rank - axis);
    BuildKrnlLoop buildInnerSoftmaxLoop(rewriter, loc, /*loopNum=*/rank - axis);

    KrnlIterateOp outerIterateOp, maxIterateOp, sumIterateOp, softmaxIterateOp;
    ArrayRef<BlockArgument> outerLoopIVs;

    if (axis != 0) {
      rewriter.setInsertionPointToStart(buildOuterLoops.getIterateBlock());
      outerLoopIVs = buildOuterLoops.getAllInductionVar();

      // Reset accumulators.
      rewriter.create<KrnlStoreOp>(loc, zero, sumOp, ArrayRef<Value>{});
      rewriter.create<KrnlStoreOp>(loc, negInfinity, maxOp, ArrayRef<Value>{});
    } else {
      // Reset accumulators.
      rewriter.create<KrnlStoreOp>(loc, zero, sumOp, ArrayRef<Value>{});
      rewriter.create<KrnlStoreOp>(loc, negInfinity, maxOp, ArrayRef<Value>{});
    }

    buildInnerMaxLoop.createDefineOp();
    buildInnerSumLoop.createDefineOp();
    buildInnerSoftmaxLoop.createDefineOp();

    for (int i = axis; i < rank; i++) {
      buildInnerMaxLoop.pushBounds(0, input, i);
      buildInnerSumLoop.pushBounds(0, input, i);
      buildInnerSoftmaxLoop.pushBounds(0, input, i);
    }

    buildInnerMaxLoop.createIterateOp();
    buildInnerSumLoop.createIterateOp();
    buildInnerSoftmaxLoop.createIterateOp();

    // Insert instructions inside the max loop.
    rewriter.setInsertionPointToStart(buildInnerMaxLoop.getIterateBlock());

    // Get induction variables.
    SmallVector<Value, 4> maxLoopIVs;
    for (auto arg : outerLoopIVs)
      maxLoopIVs.push_back(arg);
    for (auto arg : buildInnerMaxLoop.getAllInductionVar())
      maxLoopIVs.push_back(arg);

    // Compute the max value.
    Value max = rewriter.create<KrnlLoadOp>(loc, maxOp);
    Value nextMax = rewriter.create<KrnlLoadOp>(loc, input, maxLoopIVs);
    auto maxCond =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, max, nextMax);
    max = rewriter.create<SelectOp>(loc, maxCond, max, nextMax);
    rewriter.create<KrnlStoreOp>(loc, max, maxOp, ArrayRef<Value>{});

    // Get the max.
    rewriter.setInsertionPoint(
        buildInnerSumLoop.getIterateBlock()->getParentOp());
    max = rewriter.create<KrnlLoadOp>(loc, maxOp);

    // Insert instructions inside the sum loop.
    rewriter.setInsertionPointToStart(buildInnerSumLoop.getIterateBlock());

    // Get induction variables.
    SmallVector<Value, 4> sumLoopIVs;
    for (auto arg : outerLoopIVs)
      sumLoopIVs.push_back(arg);
    for (auto arg : buildInnerSumLoop.getAllInductionVar())
      sumLoopIVs.push_back(arg);

    // Sum up values.
    Value sum = rewriter.create<KrnlLoadOp>(loc, sumOp);
    Value next = rewriter.create<KrnlLoadOp>(loc, input, sumLoopIVs);
    Value sub = rewriter.create<SubFOp>(loc, next, max);
    Value exp = rewriter.create<math::ExpOp>(loc, sub);
    sum = rewriter.create<AddFOp>(loc, sum, exp);
    rewriter.create<KrnlStoreOp>(loc, sum, sumOp, ArrayRef<Value>{});
    // Store intermediate values in the result to avoid recomputation.
    rewriter.create<KrnlStoreOp>(loc, exp, alloc, sumLoopIVs);

    // Get the sum.
    rewriter.setInsertionPoint(
        buildInnerSoftmaxLoop.getIterateBlock()->getParentOp());
    sum = rewriter.create<KrnlLoadOp>(loc, sumOp);

    // Insert instructions inside the softmax loop.
    rewriter.setInsertionPointToStart(buildInnerSoftmaxLoop.getIterateBlock());

    // Get induction variables.
    SmallVector<Value, 4> softmaxLoopIVs;
    for (auto arg : outerLoopIVs)
      softmaxLoopIVs.push_back(arg);
    for (auto arg : buildInnerSoftmaxLoop.getAllInductionVar())
      softmaxLoopIVs.push_back(arg);

    // Compute softmax.
    Value expLoadedVal =
        rewriter.create<KrnlLoadOp>(loc, alloc, softmaxLoopIVs);
    Value result = rewriter.create<DivFOp>(loc, expLoadedVal, sum);
    rewriter.create<KrnlStoreOp>(loc, result, alloc, softmaxLoopIVs);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXSoftmaxOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSoftmaxOpLowering>(ctx);
}
