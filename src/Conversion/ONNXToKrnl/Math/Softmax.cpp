/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Softmax.cpp - Softmax Op ---------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
  int64_t rank = alloc.getType().cast<MemRefType>().getRank();

  // Compute the maximum value along axis.
  ValueRange maxLoops = createKrnl.defineLoops(numberOfLoops);
  createKrnl.iterateIE(maxLoops, maxLoops, Lbs, Ubs,
      [&](KrnlBuilder &createKrnl, ValueRange maxIndices) {
        MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
        IndexExprScope ieScope(createKrnl);

        // Get induction variables.
        SmallVector<Value, 4> maxLoopIVs;
        if (coerced) {
          for (auto iv : outerIndices)
            maxLoopIVs.push_back(iv);
          for (auto iv : maxIndices)
            maxLoopIVs.push_back(iv);
        } else {
          for (int64_t i = 0; i < axis; i++)
            maxLoopIVs.push_back(outerIndices[i]);
          maxLoopIVs.push_back(maxIndices[0]);
          for (int64_t i = axis + 1; i < rank; i++)
            maxLoopIVs.push_back(outerIndices[i - 1]);
        }

        Value max = create.krnl.load(maxOp, {});
        Value nextMax = create.krnl.load(input, maxLoopIVs);
        auto maxCond = create.math.sgt(max, nextMax);
        max = create.math.select(maxCond, max, nextMax);
        create.krnl.store(max, maxOp, ArrayRef<Value>{});
      });
  // Load the maximum value.
  Value max = createKrnl.load(maxOp, {});

  // Compute the sum of all values along axis.
  ValueRange sumLoops = createKrnl.defineLoops(numberOfLoops);
  createKrnl.iterateIE(sumLoops, sumLoops, Lbs, Ubs,
      [&](KrnlBuilder &createKrnl, ValueRange sumIndices) {
        MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
        IndexExprScope ieScope(createKrnl);

        // Get induction variables.
        SmallVector<Value, 4> sumLoopIVs;
        if (coerced) {
          for (auto iv : outerIndices)
            sumLoopIVs.push_back(iv);
          for (auto iv : sumIndices)
            sumLoopIVs.push_back(iv);
        } else {
          for (int64_t i = 0; i < axis; i++)
            sumLoopIVs.push_back(outerIndices[i]);
          sumLoopIVs.push_back(sumIndices[0]);
          for (int64_t i = axis + 1; i < rank; i++)
            sumLoopIVs.push_back(outerIndices[i - 1]);
        }

        Value sum = create.krnl.load(sumOp, {});
        Value next = create.krnl.load(input, sumLoopIVs);
        Value sub = create.math.sub(next, max);
        Value exp = create.math.exp(sub);
        sum = create.math.add(sum, exp);
        create.krnl.store(sum, sumOp, ArrayRef<Value>{});
        // Store intermediate values in the result to avoid
        // recomputation.
        create.krnl.store(exp, alloc, sumLoopIVs);
      });

  // Load the sum value.
  Value sum = createKrnl.load(sumOp, {});

  // Compute the softmax.
  ValueRange softmaxLoops = createKrnl.defineLoops(numberOfLoops);
  createKrnl.iterateIE(softmaxLoops, softmaxLoops, Lbs, Ubs,
      [&](KrnlBuilder &createKrnl, ValueRange softmaxIndices) {
        MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
        IndexExprScope ieScope(createKrnl);

        // Get induction variables.
        SmallVector<Value, 4> softmaxLoopIVs;
        if (coerced) {
          for (auto iv : outerIndices)
            softmaxLoopIVs.push_back(iv);
          for (auto iv : softmaxIndices)
            softmaxLoopIVs.push_back(iv);
        } else {
          for (int64_t i = 0; i < axis; i++)
            softmaxLoopIVs.push_back(outerIndices[i]);
          softmaxLoopIVs.push_back(softmaxIndices[0]);
          for (int64_t i = axis + 1; i < rank; i++)
            softmaxLoopIVs.push_back(outerIndices[i - 1]);
        }

        Value expLoadedVal = create.krnl.load(alloc, softmaxLoopIVs);
        Value result = create.math.div(expLoadedVal, sum);
        create.krnl.store(result, alloc, softmaxLoopIVs);
      });
}

static void emitInstForSoftmaxBeforeV13(ConversionPatternRewriter &rewriter,
    Location loc, Value alloc, Value input, Value sumOp, Value maxOp,
    Value zero, Value negInfinity, int64_t axis) {
  int64_t rank = alloc.getType().cast<MemRefType>().getRank();

  KrnlBuilder createKrnl(rewriter, loc);
  IndexExprScope ieScope(createKrnl);
  MemRefBoundsIndexCapture inputBounds(input);

  // Coerce the input into a 2-D tensor. `axis` will be the coercing
  // point. This coercing follows the softmax definition in ONNX:
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
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

    emitInnerLoops(createKrnl, numberOfLoops, Lbs, Ubs, {}, input, alloc, sumOp,
        maxOp, axis, /*coerced=*/true);
  } else {
    // Define outer loops.
    ValueRange outerLoops = createKrnl.defineLoops(axis);
    SmallVector<IndexExpr, 4> outerLbs(axis, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> outerUbs;
    for (int i = 0; i < axis; ++i)
      outerUbs.emplace_back(inputBounds.getDim(i));
    createKrnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs,
        [&](KrnlBuilder &createKrnl, ValueRange outerIndices) {
          IndexExprScope ieScope(createKrnl);

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
}

static void emitInstForSoftmaxV13(ConversionPatternRewriter &rewriter,
    Location loc, Value alloc, Value input, Value sumOp, Value maxOp,
    Value zero, Value negInfinity, int64_t axis) {
  int64_t rank = alloc.getType().cast<MemRefType>().getRank();

  KrnlBuilder createKrnl(rewriter, loc);
  IndexExprScope ieScope(createKrnl);
  MemRefBoundsIndexCapture inputBounds(input);

  // In opset version 13, The "axis" attribute indicates the dimension along
  // which Softmax will be performed. No need to coerce the dimensions after
  // "axis".

  // Outer loops iterate over all dimensions except axis.
  ValueRange outerLoops = createKrnl.defineLoops(rank - 1);
  SmallVector<IndexExpr, 4> outerLbs(rank - 1, LiteralIndexExpr(0));
  SmallVector<IndexExpr, 4> outerUbs;
  for (int i = 0; i < rank; ++i)
    if (i != axis)
      outerUbs.emplace_back(inputBounds.getDim(i));

  // Emit outer loops.
  createKrnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs,
      [&](KrnlBuilder &createKrnl, ValueRange outerIndices) {
        IndexExprScope ieScope(createKrnl);

        // Reset accumulators.
        createKrnl.store(zero, sumOp, ArrayRef<Value>{});
        createKrnl.store(negInfinity, maxOp, ArrayRef<Value>{});

        // Common information to create inner nested loops for axis only.
        int64_t numberOfLoops = 1;
        SmallVector<IndexExpr, 4> Lbs(numberOfLoops, LiteralIndexExpr(0));
        SmallVector<IndexExpr, 4> Ubs(numberOfLoops, inputBounds.getDim(axis));

        // Emit the inner loops.
        emitInnerLoops(createKrnl, numberOfLoops, Lbs, Ubs, outerIndices, input,
            alloc, sumOp, maxOp, axis, /*coerced=*/false);
      });
}

struct ONNXSoftmaxOpLowering : public ConversionPattern {
  ONNXSoftmaxOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSoftmaxOp::getOperationName(), 1, ctx) {}
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

    // Get opset number. Default is opset 11.
    int64_t opset = 11;
    IntegerAttr opsetAttr = op->getAttrOfType<::mlir::Attribute>("onnx_opset")
                                .dyn_cast_or_null<IntegerAttr>();
    if (opsetAttr)
      opset = opsetAttr.getValue().getSExtValue();

    auto loc = op->getLoc();
    ONNXSoftmaxOpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.input();
    // Insert an allocation and deallocation for the result of this operation.
    auto elementType = memRefType.getElementType();

    bool insertDealloc = checkInsertDealloc(op);
    Value alloc =
        (hasAllConstantDimensions(memRefType))
            ? insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc)
            : insertAllocAndDealloc(
                  memRefType, loc, rewriter, insertDealloc, input);

    // Insert allocations and deallocations for sum and max.
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    Value sumOp = insertAllocAndDealloc(scalarMemRefType, loc, rewriter, true);
    Value maxOp = insertAllocAndDealloc(scalarMemRefType, loc, rewriter, true);
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);

    MultiDialectBuilder<MathBuilder> create(rewriter, loc);
    Value negInfinity = create.math.constant(
        elementType, -std::numeric_limits<float>::infinity());

    if (opset < 13)
      // For Softmax opset < 13, `axis` is the coerced point. All dimensions
      // after `axis` will be logically coerced into a single dimension.
      emitInstForSoftmaxBeforeV13(
          rewriter, loc, alloc, input, sumOp, maxOp, zero, negInfinity, axis);
    else
      // For Softmax opset 13, `axis` attribute indicates the dimension along
      // which Softmax will be performed. No need to coerce the dimensions after
      // `axis`.
      emitInstForSoftmaxV13(
          rewriter, loc, alloc, input, sumOp, maxOp, zero, negInfinity, axis);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSoftmaxOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSoftmaxOpLowering>(typeConverter, ctx);
}
