/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- CumSum.cpp - Lowering CumSum Ops ----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX CumSum Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXCumSumOpLowering : public ConversionPattern {
  ONNXCumSumOpLowering(MLIRContext *ctx)
      : ConversionPattern(ONNXCumSumOp::getOperationName(), 1, ctx) {}

  /// We use a naive alogrithm for cumsum as follows:
  /// Assum that input is x whose shape in [n,m], and axis for cumsum is 0.
  /// ```
  /// y = x
  /// for step from 1 to log2(n):
  ///   for i range(n):
  ///     for k range(m):
  ///       if i >= 2^step:
  ///         y[i,k] = y[i - 2^(step-1),k] + y[i,k]
  ///       else:
  ///         y[i,k] = y[i,k]
  ///
  /// ```
  ///
  /// Blelloch algorithm [1] is more work-efficent. However, it is not
  /// affine-friendly, because the inner bounds depend on the outer bounds.
  ///
  /// [1] Blelloch, Guy E. 1990. "Prefix Sums and Their Applications."
  /// Technical Report CMU-CS-90-190, School of Computer Science, Carnegie
  /// Mellon University.
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXCumSumOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Builder helper.
    IndexExprScope outerScope(rewriter, loc);
    KrnlBuilder createKrnl(rewriter, loc);
    MathBuilder createMath(createKrnl);
    MemRefBuilder createMemRef(createKrnl);

    // Common information.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = memRefType.getElementType();
    Type i64Ty = rewriter.getI64Type();
    Type f32Ty = rewriter.getF32Type();

    Value X = operandAdaptor.x();
    Value axis = operandAdaptor.axis();
    MemRefBoundsIndexCapture xBounds(X);
    uint64_t rank = xBounds.getRank();
    LiteralIndexExpr zero(0);
    LiteralIndexExpr one(1);

    // Read axis.
    ArrayValueIndexCapture axisCapture(op, axis,
        getDenseElementAttributeFromConstantValue,
        loadDenseElementArrayValueAtIndex);
    SymbolIndexExpr axisIE(axisCapture.getSymbol(0));
    if (axisIE.isUndefined())
      return op->emitError("axis parameter could not be processed");

    // Insert an allocation and deallocation for the result of this operation.
    Value resMemRef;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      resMemRef =
          insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      resMemRef =
          insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc, X);

    // Get the size of dimension 'axis'.
    IndexExpr axisSize = LiteralIndexExpr(-1);
    for (uint64_t i = 0; i < rank; ++i)
      axisSize = IndexExpr::select(axisIE == i, xBounds.getDim(i), axisSize);

    // Compute log2(n), the number of steps.
    SymbolIndexExpr numberOfStep;
    if (axisSize.isLiteral()) {
      int64_t n = axisSize.getLiteral();
      float logn = std::floor(std::log2(n));
      numberOfStep = LiteralIndexExpr((int64_t)logn);
    } else {
      Value nos = rewriter.create<IndexCastOp>(loc, i64Ty, axisSize.getValue());
      nos = rewriter.create<SIToFPOp>(loc, f32Ty, nos);
      nos = createMath.log2(nos);
      nos = rewriter.create<FPToSIOp>(loc, i64Ty, nos);
      numberOfStep = SymbolIndexExpr(nos);
    }

    // Input and output have the same shape, so they shared the bounds.
    SmallVector<IndexExpr, 4> lbs(rank, zero);
    SmallVector<IndexExpr, 4> ubs;
    xBounds.getDimList(ubs);

    // Initialize the output: copy values from the input.
    ValueRange initLoopDef = createKrnl.defineLoops(rank);
    createKrnl.iterateIE(initLoopDef, initLoopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange initLoopInd) {
          Value x = createKrnl.load(X, initLoopInd);
          createKrnl.store(x, resMemRef, initLoopInd);
        });

    // Outer loop iterates over the number of steps.
    ValueRange stepLoopDef = createKrnl.defineLoops(1);
    createKrnl.iterateIE(stepLoopDef, stepLoopDef, {one}, {numberOfStep + 1},
        [&](KrnlBuilder &createKrnl, ValueRange stepLoopInd) {
          MathBuilder createMath(createKrnl);

          // Compute index offset: offset = 2^step.
          Value step = stepLoopInd[0];
          step = rewriter.create<IndexCastOp>(loc, i64Ty, step);
          step = rewriter.create<SIToFPOp>(loc, f32Ty, step);
          Value indOffset = createMath.exp2(step);
          indOffset = rewriter.create<FPToSIOp>(loc, i64Ty, indOffset);
          DimIndexExpr offset(indOffset);

          // Inner loop iterates over the output to compute sums.
          //   for i range(n):
          //     for k range(m):
          //       if i >= 2^step:
          //         y[i,k] = y[i - 2^(step-1),k] + y[i,k]
          //       else:
          //         y[i,k] = y[i,k]
          ValueRange sumLoopDef = createKrnl.defineLoops(rank);
          createKrnl.iterateIE(sumLoopDef, sumLoopDef, lbs, ubs,
              [&](KrnlBuilder &createKrnl, ValueRange sumLoopInd) {
                IndexExprScope ieScope(createKrnl);
                MathBuilder createMath(createKrnl);
                SymbolIndexExpr offsetIE(offset);
                // Load y[i,k].
                Value y1 = createKrnl.load(resMemRef, sumLoopInd);
                // Load y[i - 2^(step-1),k].
                SmallVector<IndexExpr, 4> loopInd;
                for (uint64_t i = 0; i < rank; ++i) {
                  DimIndexExpr iIE(sumLoopInd[i]);
                  IndexExpr newiIE =
                      iIE.select(iIE == axisIE, iIE - offsetIE, iIE);
                  IndexExpr finaliIE = newiIE.selectOrSelf(newiIE < 0, iIE);
                  loopInd.emplace_back(iIE);
                }
                Value y2 = createKrnl.loadIE(resMemRef, loopInd);
                Value zeroVal = emitConstantOp(rewriter, loc, elementType, 0);
                createKrnl.store(y1, resMemRef, sumLoopInd);
                createKrnl.store(y2, resMemRef, sumLoopInd);
              });
        });

    rewriter.replaceOp(op, resMemRef);
    return success();
  }
};

void populateLoweringONNXCumSumOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXCumSumOpLowering>(ctx);
}
