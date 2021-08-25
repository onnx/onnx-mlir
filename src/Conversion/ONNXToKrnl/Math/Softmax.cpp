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

static void emitInnerLoops(KrnlBuilder &createKrnl, int64_t numberOfLoops,
    SmallVectorImpl<IndexExpr> &Lbs, SmallVectorImpl<IndexExpr> &Ubs,
    ValueRange outerIndices, Value input, Value alloc, Value sumOp, Value maxOp,
    int64_t axis, bool coerced = true) {

  // Compute the maximum value along axis.
  ValueRange maxLoops = createKrnl.defineLoops(numberOfLoops);
  createKrnl.iterateIE(maxLoops, maxLoops, Lbs, Ubs, {},
      [&](KrnlBuilder &createKrnl, ValueRange args) {
        MathBuilder createMath(createKrnl);
        IndexExprScope ieScope(createKrnl);
        ValueRange maxIndices = createKrnl.getInductionVarValue(maxLoops);

        // Get induction variables.
        SmallVector<Value, 4> maxLoopIVs;
        for (auto iv : outerIndices)
          maxLoopIVs.push_back(iv);
        for (auto iv : maxIndices)
          maxLoopIVs.push_back(iv);

        Value max = createKrnl.load(maxOp, {});
        Value nextMax = createKrnl.load(input, maxLoopIVs);
        auto maxCond = createMath.sgt(max, nextMax);
        max = createMath.select(maxCond, max, nextMax);
        createKrnl.store(max, maxOp, ArrayRef<Value>{});
      });
  // Load the maximum value.
  Value max = createKrnl.load(maxOp, {});

  // Compute the sum of all values along axis.
  ValueRange sumLoops = createKrnl.defineLoops(numberOfLoops);
  createKrnl.iterateIE(sumLoops, sumLoops, Lbs, Ubs, {},
      [&](KrnlBuilder &createKrnl, ValueRange args) {
        MathBuilder createMath(createKrnl);
        IndexExprScope ieScope(createKrnl);
        ValueRange sumIndices = createKrnl.getInductionVarValue(sumLoops);

        // Get induction variables.
        SmallVector<Value, 4> sumLoopIVs;
        for (auto iv : outerIndices)
          sumLoopIVs.push_back(iv);
        for (auto iv : sumIndices)
          sumLoopIVs.push_back(iv);

        Value sum = createKrnl.load(sumOp, {});
        Value next = createKrnl.load(input, sumLoopIVs);
        Value sub = createMath.sub(next, max);
        Value exp = createMath.exp(sub);
        sum = createMath.add(sum, exp);
        createKrnl.store(sum, sumOp, ArrayRef<Value>{});
        // Store intermediate values in the result to avoid
        // recomputation.
        createKrnl.store(exp, alloc, sumLoopIVs);
      });

  // Load the sum value.
  Value sum = createKrnl.load(sumOp, {});

  // Compute the softmax.
  ValueRange softmaxLoops = createKrnl.defineLoops(numberOfLoops);
  createKrnl.iterateIE(softmaxLoops, softmaxLoops, Lbs, Ubs, {},
      [&](KrnlBuilder &createKrnl, ValueRange args) {
        MathBuilder createMath(createKrnl);
        IndexExprScope ieScope(createKrnl);
        ValueRange softmaxIndices =
            createKrnl.getInductionVarValue(softmaxLoops);

        // Get induction variables.
        SmallVector<Value, 4> softmaxLoopIVs;
        for (auto iv : outerIndices)
          softmaxLoopIVs.push_back(iv);
        for (auto iv : softmaxIndices)
          softmaxLoopIVs.push_back(iv);

        Value expLoadedVal = createKrnl.load(alloc, softmaxLoopIVs);
        Value result = createMath.divf(expLoadedVal, sum);
        createKrnl.store(result, alloc, softmaxLoopIVs);
      });
}

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

    // Insert allocations and deallocations for sum and max.
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    Value sumOp = insertAllocAndDealloc(scalarMemRefType, loc, rewriter, true);
    Value maxOp = insertAllocAndDealloc(scalarMemRefType, loc, rewriter, true);
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    Value negInfinity = rewriter.create<ConstantOp>(loc,
        FloatAttr::get(elementType, -std::numeric_limits<float>::infinity()));

    ImplicitLocOpBuilder ilob(loc, rewriter);
    KrnlBuilder createKrnl(ilob);
    IndexExprScope ieScope(createKrnl);
    MemRefBoundsIndexCapture inputBounds(input);

    // Coerce the input into a 2-D tensor. `axis` will be the coercing
    // point. This coercing follows the softmax definition in ONNX:
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
    // Here, we create an outer loop and inner loop for handling the two
    // dimensions. The outer loop is only created once `axis` is not
    // zero.
    if (axis == 0) {
      // There is no need having outer loops.
      // Reset accumulators.
      createKrnl.store(zero, sumOp, ArrayRef<Value>{});
      createKrnl.store(negInfinity, maxOp, ArrayRef<Value>{});

      // Common information to create nested loops.
      int64_t numberOfLoops = rank;
      SmallVector<IndexExpr, 4> Lbs(numberOfLoops, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> Ubs;
      inputBounds.getDimList(Ubs);

      emitInnerLoops(createKrnl, numberOfLoops, Lbs, Ubs, {}, input, alloc,
          sumOp, maxOp, axis, /*coerced=*/true);
    } else {
      // Define outer loops.
      ValueRange outerLoops = createKrnl.defineLoops(axis);
      SmallVector<IndexExpr, 4> outerLbs(axis, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> outerUbs;
      for (int i = 0; i < axis; ++i)
        outerUbs.emplace_back(inputBounds.getDim(i));
      createKrnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs, {},
          [&](KrnlBuilder &createKrnl, ValueRange args) {
            IndexExprScope ieScope(createKrnl);
            ValueRange outerIndices =
                createKrnl.getInductionVarValue(outerLoops);

            // Reset accumulators.
            createKrnl.store(zero, sumOp, ArrayRef<Value>{});
            createKrnl.store(negInfinity, maxOp, ArrayRef<Value>{});

            // Common information to create inner nested loops.
            int64_t numberOfLoops = rank - axis;
            SmallVector<IndexExpr, 4> Lbs(numberOfLoops, LiteralIndexExpr(0));
            SmallVector<IndexExpr, 4> Ubs;
            for (int i = axis; i < rank; ++i)
              Ubs.emplace_back(inputBounds.getDim(i));

            // Emit the inner loops.
            emitInnerLoops(createKrnl, numberOfLoops, Lbs, Ubs, outerIndices,
                input, alloc, sumOp, maxOp, axis, /*coerced=*/true);
          });
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSoftmaxOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSoftmaxOpLowering>(ctx);
}
