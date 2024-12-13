/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Softmax.cpp - Softmax Op ---------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include <src/Dialect/ONNX/ONNXOps.hpp>

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;
namespace onnx_mlir {

// TODO: may consider exploiting SIMD.

static void emitInnerLoops(KrnlBuilder &createKrnl, int64_t numberOfLoops,
    SmallVectorImpl<IndexExpr> &Lbs, SmallVectorImpl<IndexExpr> &Ubs,
    ValueRange outerIndices, Value input, Value alloc, Value zero,
    Value negInfinity, int64_t axis, bool coerced = true) {
  int64_t rank = mlir::cast<MemRefType>(alloc.getType()).getRank();

  ValueRange maxInits = ValueRange(negInfinity);
  // Compute the maximum value along axis.
  ValueRange maxLoops = createKrnl.defineLoops(numberOfLoops);
  auto maxLoop = createKrnl.iterateIE(maxLoops, maxLoops, Lbs, Ubs, maxInits,
      [&](const KrnlBuilder &createKrnl, ValueRange maxIndices,
          ValueRange iterArgs) {
        // Get last argument for the iterate body.
        Value iterArg = iterArgs.back();

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

        Value max = iterArg;
        Value nextMax = create.krnl.load(input, maxLoopIVs);
        max = create.math.max(max, nextMax);
        create.krnl.yield(max);
      });
  // Get the maximum value.
  Value max = maxLoop.getResult(0);

  ValueRange sumInits = ValueRange(zero);
  // Compute the sum of all values along axis.
  ValueRange sumLoops = createKrnl.defineLoops(numberOfLoops);
  auto sumLoop = createKrnl.iterateIE(sumLoops, sumLoops, Lbs, Ubs, sumInits,
      [&](const KrnlBuilder &createKrnl, ValueRange sumIndices,
          ValueRange iterArgs) {
        // Get last argument for the iterate body.
        Value iterArg = iterArgs.back();

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

        Value sum = iterArg;
        Value next = create.krnl.load(input, sumLoopIVs);
        Value sub = create.math.sub(next, max);
        Value exp = create.math.exp(sub);
        sum = create.math.add(sum, exp);
        // Store intermediate values in the result to avoid
        // recomputation.
        create.krnl.store(exp, alloc, sumLoopIVs);
        create.krnl.yield(sum);
      });

  // Load the sum value.
  Value sum = sumLoop.getResult(0);

  // Compute the softmax.
  ValueRange softmaxLoops = createKrnl.defineLoops(numberOfLoops);
  createKrnl.iterateIE(softmaxLoops, softmaxLoops, Lbs, Ubs,
      [&](const KrnlBuilder &createKrnl, ValueRange softmaxIndices) {
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
void emitInstForSoftmax(ConversionPatternRewriter &rewriter, Operation *op,
    Location loc, Value alloc, Value input, Value zero, Value negInfinity,
    int64_t axis, bool enableParallel) = delete;

// For Softmax opset < 13, `axis` is the coerced point. All dimensions
// after `axis` will be logically coerced into a single dimension.
template <>
void emitInstForSoftmax<ONNXSoftmaxV11Op>(ConversionPatternRewriter &rewriter,
    Operation *op, Location loc, Value alloc, Value input, Value zero,
    Value negInfinity, int64_t axis, bool enableParallel) {
  int64_t rank = mlir::cast<MemRefType>(alloc.getType()).getRank();

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
    assert(!enableParallel && "only outer loop parallelism at this time");
    // There is no need having outer loops.

    // Common information to create nested loops.
    int64_t numberOfLoops = rank;
    SmallVector<IndexExpr, 4> Lbs(numberOfLoops, zeroIE);
    SmallVector<IndexExpr, 4> Ubs;
    create.krnlIE.getShapeAsDims(input, Ubs);

    emitInnerLoops(create.krnl, numberOfLoops, Lbs, Ubs, {}, input, alloc, zero,
        negInfinity, axis, /*coerced=*/true);
  } else {
    // Define outer loops.
    ValueRange outerLoops = create.krnl.defineLoops(axis);
    SmallVector<IndexExpr, 4> outerLbs(axis, zeroIE);
    SmallVector<IndexExpr, 4> outerUbs;
    for (int i = 0; i < axis; ++i)
      outerUbs.emplace_back(create.krnlIE.getShapeAsDim(input, i));
    if (enableParallel) {
      assert(axis > 0 && "bad assumption");
      int64_t parId;
      if (findSuitableParallelDimension(outerLbs, outerUbs, 0, 1, parId,
              /*min iter for going parallel*/ 4)) {
        create.krnl.parallel(outerLoops[0]);
        onnxToKrnlParallelReport(
            op, true, 0, outerLbs[0], outerUbs[0], "softmax v11");
      } else {
        onnxToKrnlParallelReport(
            op, false, 0, outerLbs[0], outerUbs[0], "softmax v11");
      }
    }
    create.krnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs,
        [&](const KrnlBuilder &ck, ValueRange outerIndices) {
          MultiDialectBuilder<MemRefBuilder, KrnlBuilder,
              IndexExprBuilderForKrnl>
              create(ck);
          IndexExprScope ieScope(ck);

          // Common information to create inner nested loops.
          int64_t numberOfLoops = rank - axis;
          SmallVector<IndexExpr, 4> Lbs(numberOfLoops, zeroIE);
          SmallVector<IndexExpr, 4> Ubs;
          for (int i = axis; i < rank; ++i)
            Ubs.emplace_back(create.krnlIE.getShapeAsDim(input, i));

          // Emit the inner loops.
          emitInnerLoops(create.krnl, numberOfLoops, Lbs, Ubs, outerIndices,
              input, alloc, zero, negInfinity, axis, /*coerced=*/true);
        });
  }
}

// For Softmax opset 13, `axis` attribute indicates the dimension along
// which Softmax will be performed. No need to coerce the dimensions after
// `axis`.
template <>
void emitInstForSoftmax<ONNXSoftmaxOp>(ConversionPatternRewriter &rewriter,
    Operation *op, Location loc, Value alloc, Value input, Value zero,
    Value negInfinity, int64_t axis, bool enableParallel) {
  int64_t rank = mlir::cast<MemRefType>(alloc.getType()).getRank();

  MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
      rewriter, loc);
  IndexExprScope ieScope(create.krnl);
  LiteralIndexExpr zeroIE(0);

  // Parallel only if output is not a scalar.
  if (rank - 1 == 0)
    enableParallel = false;

  // Outer loops iterate over all dimensions except axis.
  ValueRange outerLoops = create.krnl.defineLoops(rank - 1);
  SmallVector<IndexExpr, 4> outerLbs(rank - 1, zeroIE);
  SmallVector<IndexExpr, 4> outerUbs;
  for (int i = 0; i < rank; ++i)
    if (i != axis)
      outerUbs.emplace_back(create.krnlIE.getShapeAsDim(input, i));

  if (enableParallel) {
    int64_t parId;
    if (findSuitableParallelDimension(outerLbs, outerUbs, 0, 1, parId,
            /*min iter for going parallel*/ 4)) {
      create.krnl.parallel(outerLoops[0]);
      onnxToKrnlParallelReport(
          op, true, 0, outerLbs[0], outerUbs[0], "softmax");
    } else {
      onnxToKrnlParallelReport(op, false, 0, outerLbs[0], outerUbs[0],
          "not enough work for softmax");
    }
  }

  // Emit outer loops.
  create.krnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs,
      [&](const KrnlBuilder &ck, ValueRange outerIndices) {
        MultiDialectBuilder<MemRefBuilder, KrnlBuilder, IndexExprBuilderForKrnl>
            create(ck);
        IndexExprScope ieScope(ck);

        // Common information to create inner nested loops for axis only.
        int64_t numberOfLoops = 1;
        SmallVector<IndexExpr, 4> Lbs(numberOfLoops, zeroIE);
        SmallVector<IndexExpr, 4> Ubs(
            numberOfLoops, create.krnlIE.getShapeAsDim(input, axis));

        // Emit the inner loops.
        emitInnerLoops(create.krnl, numberOfLoops, Lbs, Ubs, outerIndices,
            input, alloc, zero, negInfinity, axis, /*coerced=*/false);
      });
}

template <typename SoftmaxOp>
struct ONNXSoftmaxLowering : public OpConversionPattern<SoftmaxOp> {
  ONNXSoftmaxLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern<SoftmaxOp>(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXSoftmaxOp::getOperationName());
  }

  using OpAdaptor = typename SoftmaxOp::Adaptor;
  bool enableParallel = false;

  LogicalResult matchAndRewrite(SoftmaxOp softmaxOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // softmax(x) = let max_x = max(x) in
    //                let exp_x = exp(x - max_x) in
    //                  let sum = sum(exp_x) in
    //                    exp_x / sum

    Operation *op = softmaxOp.getOperation();
    Location loc = ONNXLoc<SoftmaxOp>(op);
    Value input = adaptor.getInput();

    // Convert the output type to MemRefType.
    Type convertedType =
        this->typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

    int64_t rank = memRefType.getRank();
    int64_t axis = adaptor.getAxis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);
    // Since we parallelize the outer loops only at this time, disable the case
    // where there is no outer loops at all.
    bool enableParallelLocal = enableParallel;
    if (axis <= 0)
      enableParallelLocal = false;

    // Insert an allocation and deallocation for the result of this operation.
    Type elementType = memRefType.getElementType();
    MultiDialectBuilder<MemRefBuilder, MathBuilder> create(rewriter, loc);
    Value alloc = create.mem.alignedAlloc(input, memRefType);

    Value zero = create.math.constant(elementType, 0);
    Value negInfinity = create.math.constant(
        elementType, -std::numeric_limits<float>::infinity());

    emitInstForSoftmax<SoftmaxOp>(rewriter, op, loc, alloc, input, zero,
        negInfinity, axis, enableParallelLocal);

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXSoftmaxOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXSoftmaxLowering<ONNXSoftmaxOp>,
      ONNXSoftmaxLowering<ONNXSoftmaxV11Op>>(
      typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
