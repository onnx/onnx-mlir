/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Hardmax.cpp - Hardmax Op ---------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

namespace onnx_mlir {

/// Returns the indices of the maximum values along a given axis.
static Value emitArgmax(ConversionPatternRewriter &rewriter, Location loc,
    Value input, int64_t axis) {
  MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
      MemRefBuilder>
      create(rewriter, loc);
  IndexExprScope scope(create.krnl);

  MemRefType memRefType = mlir::cast<MemRefType>(input.getType());
  Type indexType = rewriter.getIndexType();
  int64_t rank = memRefType.getRank();
  Value zero = create.math.constantIndex(0);

  SmallVector<IndexExpr, 4> inputUBS;
  create.krnlIE.getShapeAsDims(input, inputUBS);

  // Allocate and initialize the result.
  // Th result has the same shape as the input except the axis dimension is 1.
  SmallVector<IndexExpr, 4> outputUBS(inputUBS);
  outputUBS[axis] = LitIE(1);
  SmallVector<int64_t, 4> outputShape;
  for (const IndexExpr &dim : outputUBS)
    outputShape.push_back(
        dim.isLiteral() ? dim.getLiteral() : ShapedType::kDynamic);
  Value resMemRef = create.mem.alignedAlloc(
      MemRefType::get(outputShape, indexType), outputUBS);
  create.krnl.memset(resMemRef, zero);

  ValueRange loopDef = create.krnl.defineLoops(rank);
  SmallVector<IndexExpr> lbs(rank, LitIE(0));
  create.krnl.iterateIE(loopDef, loopDef, lbs, inputUBS,
      [&](const KrnlBuilder &createKrnl, ValueRange inputLoopInd) {
        MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder> create(
            createKrnl);
        // Load the index of the current max value.
        SmallVector<Value> resLoopInd(inputLoopInd);
        resLoopInd[axis] = zero;
        Value maxInd = create.krnl.load(resMemRef, resLoopInd);

        // Load the current max value.
        SmallVector<Value> maxLoopInd(inputLoopInd);
        maxLoopInd[axis] = maxInd;
        Value maxValue = create.krnl.load(input, maxLoopInd);
        // Load a new value.
        Value next = create.krnl.load(input, inputLoopInd);

        // Compare and update the index for the maximum value.
        Value gt = create.math.sgt(next, maxValue);
        create.scf.ifThenElse(gt, [&](const SCFBuilder &createSCF) {
          KrnlBuilder createKrnl(createSCF);
          createKrnl.store(inputLoopInd[axis], resMemRef, resLoopInd);
        });
      });

  return resMemRef;
}

struct ONNXHardmaxOpLowering : public OpConversionPattern<ONNXHardmaxOp> {
  ONNXHardmaxOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}
  LogicalResult matchAndRewrite(ONNXHardmaxOp hardmaxOp,
      ONNXHardmaxOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = hardmaxOp.getOperation();
    Location loc = ONNXLoc<ONNXHardmaxOp>(op);
    Value input = adaptor.getInput();

    MultiDialectBuilder<MathBuilder, KrnlBuilder, IndexExprBuilderForKrnl,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnl);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

    Type elementType = memRefType.getElementType();
    Value zero = create.math.constantIndex(0);

    int64_t rank = memRefType.getRank();
    int64_t axis = llvm::dyn_cast<ONNXHardmaxOp>(op).getAxis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);

    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(input, ubs);

    // Insert an allocation and deallocation for the result of this operation.
    Value resMemRef = create.mem.alignedAlloc(memRefType, ubs);

    // Compute argmax.
    Value argmax = emitArgmax(rewriter, loc, input, axis);

    // Produce the final result.
    // Set value to 1 if index is argmax. Otherwise, 0.
    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr> lbs(rank, LitIE(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder> create(
              createKrnl);
          // Load the index of the current max value.
          SmallVector<Value> maxLoopInd(loopInd);
          maxLoopInd[axis] = zero;
          Value maxInd = create.krnl.load(argmax, maxLoopInd);

          // Set value to 1 if the index is argmax. Otherwise, 0.
          Value eq = create.math.eq(maxInd, loopInd[axis]);
          create.scf.ifThenElse(
              eq, /*then*/
              [&](const SCFBuilder &createSCF) {
                MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createSCF);
                Value one = create.math.constant(elementType, 1);
                create.krnl.store(one, resMemRef, loopInd);
              },
              /*else*/
              [&](const SCFBuilder &createSCF) {
                MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createSCF);
                Value zero = create.math.constant(elementType, 0);
                create.krnl.store(zero, resMemRef, loopInd);
              });
        });

    rewriter.replaceOp(op, resMemRef);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXHardmaxOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXHardmaxOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
