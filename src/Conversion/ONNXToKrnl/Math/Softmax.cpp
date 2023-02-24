/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Softmax.cpp - Softmax Op ---------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include <src/Dialect/ONNX/ONNXOps.hpp>

using namespace mlir;

namespace onnx_mlir {

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

template <typename T>
void emitInstForSoftmax(ConversionPatternRewriter &rewriter, Location loc,
    Value alloc, Value input, Value sumOp, Value maxOp, Value zero,
    Value negInfinity, int64_t axis) = delete;

// For Softmax opset < 13, `axis` is the coerced point. All dimensions
// after `axis` will be logically coerced into a single dimension.
template <>
void emitInstForSoftmax<ONNXSoftmaxV11Op>(ConversionPatternRewriter &rewriter,
    Location loc, Value alloc, Value input, Value sumOp, Value maxOp,
    Value zero, Value negInfinity, int64_t axis) {
  int64_t rank = alloc.getType().cast<MemRefType>().getRank();

  MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
      rewriter, loc);
  IndexExprScope ieScope(create.krnl);
  LiteralIndexExpr zeroIE(0);

  // Coerce the input into a 2-D tensor. `axis` will be the coercing
  // point. This coercing follows the softmax definition in ONNX:
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
  // Here, we create an outer loop and inner loop for handling the two
  // dimensions. The outer loop is only created once `axis` is not
  // zero.
  if (axis == 0) {
    // There is no need having outer loops.
    // Reset accumulators.
    create.krnl.store(zero, sumOp, ArrayRef<Value>{});
    create.krnl.store(negInfinity, maxOp, ArrayRef<Value>{});

    // Common information to create nested loops.
    int64_t numberOfLoops = rank;
    SmallVector<IndexExpr, 4> Lbs(numberOfLoops, zeroIE);
    SmallVector<IndexExpr, 4> Ubs;
    create.krnlIE.getShapeAsDims(input, Ubs);

    emitInnerLoops(create.krnl, numberOfLoops, Lbs, Ubs, {}, input, alloc,
        sumOp, maxOp, axis, /*coerced=*/true);
  } else {
    // Define outer loops.
    ValueRange outerLoops = create.krnl.defineLoops(axis);
    SmallVector<IndexExpr, 4> outerLbs(axis, zeroIE);
    SmallVector<IndexExpr, 4> outerUbs;
    for (int i = 0; i < axis; ++i)
      outerUbs.emplace_back(create.krnlIE.getShapeAsDim(input, i));
    create.krnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs,
        [&](KrnlBuilder &ck, ValueRange outerIndices) {
          MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(ck);
          IndexExprScope ieScope(ck);

          // Reset accumulators.
          create.krnl.store(zero, sumOp, ArrayRef<Value>{});
          create.krnl.store(negInfinity, maxOp, ArrayRef<Value>{});

          // Common information to create inner nested loops.
          int64_t numberOfLoops = rank - axis;
          SmallVector<IndexExpr, 4> Lbs(numberOfLoops, zeroIE);
          SmallVector<IndexExpr, 4> Ubs;
          for (int i = axis; i < rank; ++i)
            Ubs.emplace_back(create.krnlIE.getShapeAsDim(input, i));

          // Emit the inner loops.
          emitInnerLoops(create.krnl, numberOfLoops, Lbs, Ubs, outerIndices,
              input, alloc, sumOp, maxOp, axis, /*coerced=*/true);
        });
  }
}

// For Softmax opset 13, `axis` attribute indicates the dimension along
// which Softmax will be performed. No need to coerce the dimensions after
// `axis`.
template <>
void emitInstForSoftmax<ONNXSoftmaxOp>(ConversionPatternRewriter &rewriter,
    Location loc, Value alloc, Value input, Value sumOp, Value maxOp,
    Value zero, Value negInfinity, int64_t axis) {
  int64_t rank = alloc.getType().cast<MemRefType>().getRank();

  MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
      rewriter, loc);
  IndexExprScope ieScope(create.krnl);
  LiteralIndexExpr zeroIE(0);

  // Outer loops iterate over all dimensions except axis.
  ValueRange outerLoops = create.krnl.defineLoops(rank - 1);
  SmallVector<IndexExpr, 4> outerLbs(rank - 1, zeroIE);
  SmallVector<IndexExpr, 4> outerUbs;
  for (int i = 0; i < rank; ++i)
    if (i != axis)
      outerUbs.emplace_back(create.krnlIE.getShapeAsDim(input, i));

  // Emit outer loops.
  create.krnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs,
      [&](KrnlBuilder &ck, ValueRange outerIndices) {
        MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(ck);
        IndexExprScope ieScope(ck);

        // Reset accumulators.
        create.krnl.store(zero, sumOp, ArrayRef<Value>{});
        create.krnl.store(negInfinity, maxOp, ArrayRef<Value>{});

        // Common information to create inner nested loops for axis only.
        int64_t numberOfLoops = 1;
        SmallVector<IndexExpr, 4> Lbs(numberOfLoops, zeroIE);
        SmallVector<IndexExpr, 4> Ubs(
            numberOfLoops, create.krnlIE.getShapeAsDim(input, axis));

        // Emit the inner loops.
        emitInnerLoops(create.krnl, numberOfLoops, Lbs, Ubs, outerIndices,
            input, alloc, sumOp, maxOp, axis, /*coerced=*/false);
      });
}

template <typename SoftmaxOp>
struct ONNXSoftmaxLowering : public ConversionPattern {
  ONNXSoftmaxLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, SoftmaxOp::getOperationName(), 1, ctx) {}
  using OpAdaptor = typename SoftmaxOp::Adaptor;
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // softmax(x) = let max_x = max(x) in
    //                let exp_x = exp(x - max_x) in
    //                  let sum = sum(exp_x) in
    //                    exp_x / sum

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();

    int64_t rank = memRefType.getRank();
    int64_t axis = llvm::dyn_cast<SoftmaxOp>(op).getAxis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);

    Location loc = op->getLoc();
    OpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.getInput();
    // Insert an allocation and deallocation for the result of this operation.
    Type elementType = memRefType.getElementType();
    MultiDialectBuilder<MemRefBuilder, MathBuilder> create(rewriter, loc);
    Value alloc = create.mem.alignedAlloc(input, memRefType);

    // Insert allocations and deallocations for sum and max.
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    Value sumOp = create.mem.alignedAlloc(scalarMemRefType);
    Value maxOp = create.mem.alignedAlloc(scalarMemRefType);

    Value zero = create.math.constant(elementType, 0);
    Value negInfinity = create.math.constant(
        elementType, -std::numeric_limits<float>::infinity());

    emitInstForSoftmax<SoftmaxOp>(
        rewriter, loc, alloc, input, sumOp, maxOp, zero, negInfinity, axis);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSoftmaxOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSoftmaxLowering<ONNXSoftmaxOp>,
      ONNXSoftmaxLowering<ONNXSoftmaxV11Op>>(typeConverter, ctx);
}

} // namespace onnx_mlir
