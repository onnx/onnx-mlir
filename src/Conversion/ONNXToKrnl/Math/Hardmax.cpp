/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Hardmax.cpp - Hardmax Op ---------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
  IndexExprScope scope(create.krnl);

  MemRefType memRefType = input.getType().cast<MemRefType>();
  Type indexType = rewriter.getIndexType();
  int64_t rank = memRefType.getRank();
  Value zero = create.math.constantIndex(0);

  MemRefBoundsIndexCapture inputBounds(input);
  SmallVector<IndexExpr, 4> inputUBS;
  inputBounds.getDimList(inputUBS);

  // Allocate and initialize the result.
  // Th result has the same shape as the input except the axis dimension is 1.
  SmallVector<IndexExpr, 4> outputUBS(inputUBS);
  outputUBS[axis] = LiteralIndexExpr(1);
  SmallVector<int64_t, 4> outputShape;
  for (const IndexExpr &dim : outputUBS)
    outputShape.push_back(dim.isLiteral() ? dim.getLiteral() : -1);
  Value resMemRef = insertAllocAndDeallocSimple(rewriter, nullptr,
      MemRefType::get(outputShape, indexType), loc, outputUBS,
      /*insertDealloc=*/true);
  create.krnl.memset(resMemRef, zero);

  ValueRange loopDef = create.krnl.defineLoops(rank);
  SmallVector<IndexExpr> lbs(rank, LiteralIndexExpr(0));
  create.krnl.iterateIE(loopDef, loopDef, lbs, inputUBS,
      [&](KrnlBuilder &createKrnl, ValueRange inputLoopInd) {
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
        create.scf.ifThenElse(gt, [&](SCFBuilder &createSCF) {
          KrnlBuilder createKrnl(createSCF);
          createKrnl.store(inputLoopInd[axis], resMemRef, resLoopInd);
        });
      });

  return resMemRef;
}

struct ONNXHardmaxOpLowering : public ConversionPattern {
  ONNXHardmaxOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXHardmaxOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    MultiDialectBuilder<MathBuilder, KrnlBuilder> create(rewriter, loc);
    IndexExprScope scope(create.krnl);

    ONNXHardmaxOpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.input();

    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());
    auto elementType = memRefType.getElementType();
    Value zero = create.math.constantIndex(0);

    int64_t rank = memRefType.getRank();
    int64_t axis = llvm::dyn_cast<ONNXHardmaxOp>(op).axis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);

    MemRefBoundsIndexCapture inputBounds(input);
    SmallVector<IndexExpr, 4> ubs;
    inputBounds.getDimList(ubs);

    // Insert an allocation and deallocation for the result of this operation.
    bool insertDealloc = checkInsertDealloc(op);
    Value resMemRef = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, ubs, insertDealloc);

    // Compute argmax.
    Value argmax = emitArgmax(rewriter, loc, input, axis);

    // Produce the final result.
    // Set value to 1 if index is argmax. Otherwise, 0.
    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr> lbs(rank, LiteralIndexExpr(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
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
              [&](SCFBuilder &createSCF) {
                MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createSCF);
                Value one = create.math.constant(elementType, 1);
                create.krnl.store(one, resMemRef, loopInd);
              },
              /*else*/
              [&](SCFBuilder &createSCF) {
                MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createSCF);
                Value zero = create.math.constant(elementType, 0);
                create.krnl.store(zero, resMemRef, loopInd);
              });
        });

    rewriter.replaceOp(op, resMemRef);
    return success();
  }
};

void populateLoweringONNXHardmaxOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXHardmaxOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
