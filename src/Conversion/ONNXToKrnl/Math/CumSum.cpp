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
  /// We double-buffer the output to avoid intermediate result being overwritten
  /// by multiple threads.
  /// ```
  /// buf = x
  /// for step from 0 to log2(n) + 1:
  ///   for i range(n):
  ///     for k range(m):
  ///       if i >= 2^step:
  ///         y[i,k] = buf[i - 2^step,k] + buf[i,k]
  ///       else:
  ///         y[i,k] = buf[i,k]
  ///   buf = y
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
    ONNXCumSumOp csOp = llvm::cast<ONNXCumSumOp>(op);
    ONNXCumSumOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Builder helper.
    IndexExprScope outerScope(&rewriter, loc);
    KrnlBuilder createKrnl(rewriter, loc);
    MathBuilder createMath(createKrnl);
    MemRefBuilder createMemRef(createKrnl);

    // Common information.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = memRefType.getElementType();
    Type i1Ty = rewriter.getI1Type();
    Type i64Ty = rewriter.getI64Type();
    Type f32Ty = rewriter.getF32Type();
    Type indexTy = rewriter.getIndexType();

    Value X = operandAdaptor.x();
    Value axis = operandAdaptor.axis();
    bool exclusive = csOp.exclusive() == 1;
    bool reverse = csOp.reverse() == 1;
    MemRefBoundsIndexCapture xBounds(X);
    uint64_t rank = xBounds.getRank();
    LiteralIndexExpr zero(0);

    // Read axis.
    ArrayValueIndexCapture axisCapture(axis,
        getDenseElementAttributeFromConstantValue,
        loadDenseElementArrayValueAtIndex);
    IndexExpr axisIE(axisCapture.getSymbol(0));
    if (axisIE.isUndefined())
      return op->emitError("axis parameter could not be processed");
    axisIE = axisIE.selectOrSelf(axisIE < 0, axisIE + LiteralIndexExpr(rank));

    // Insert an allocation and deallocation for the result of this operation.
    Value resMemRef, bufMemRef;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType)) {
      resMemRef =
          insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
      bufMemRef = insertAllocAndDealloc(memRefType, loc, rewriter, true);
    } else {
      resMemRef =
          insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc, X);
      bufMemRef = insertAllocAndDealloc(memRefType, loc, rewriter, true, X);
    }

    // Get the size of dimension 'axis'.
    IndexExpr axisSize = LiteralIndexExpr(-1);
    for (uint64_t i = 0; i < rank; ++i)
      axisSize = IndexExpr::select(axisIE == i, xBounds.getDim(i), axisSize);

    // Compute log2(n), the number of steps.
    IndexExpr numberOfStep;
    if (axisSize.isLiteral()) {
      int64_t n = axisSize.getLiteral();
      int64_t logn = (int64_t)std::log2(n);
      numberOfStep = LiteralIndexExpr(logn);
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

    // Initialize the temporary buffer: copy values from the input.
    ValueRange initLoopDef = createKrnl.defineLoops(rank);
    createKrnl.iterateIE(initLoopDef, initLoopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange initLoopInd) {
          if (!exclusive) {
            Value x = createKrnl.load(X, initLoopInd);
            createKrnl.store(x, bufMemRef, initLoopInd);
          } else {
            // Exclusive mode is equivalent to shift all elements right (left if
            // reversed) and set the first element (the last element if
            // reversed) to 0.
            //
            // For example, doing exclusive mode on the input:
            //   input = [2, 3, 4]
            // is equivalent to doing non-exclusive mode on:
            //   new_input = [0, 2, 3]
            // or
            //   new_input = [3, 4, 0] if reversed.
            MathBuilder createMath(createKrnl);
            SymbolIndexExpr axis(axisIE);

            SmallVector<Value, 4> loopInd;
            Value offsetOne = emitConstantOp(rewriter, loc, indexTy, 1);
            Value shiftOrSet0 = emitConstantOp(rewriter, loc, i1Ty, 0);
            for (uint64_t r = 0; r < rank; ++r) {
              Value iVal = initLoopInd[r];
              Value rVal = emitConstantOp(rewriter, loc, indexTy, r);
              // Whether we are in the reduction axis.
              Value isAxis = createMath.eq(rVal, axis.getValue());
              // Whether the current element's index is in scope.
              Value isInScope;
              if (reverse) {
                Value x = createMath.add(iVal, offsetOne);
                isInScope = createMath.slt(x, xBounds.getDim(r).getValue());
              } else {
                isInScope = createMath.sge(iVal, offsetOne);
              }
              Value ok = createMath._and(isAxis, isInScope);
              shiftOrSet0 = createMath._or(ok, shiftOrSet0);

              Value iOffset;
              if (reverse)
                iOffset = createMath.add(iVal, offsetOne);
              else
                iOffset = createMath.sub(iVal, offsetOne);
              Value accessIndex = createMath.select(ok, iOffset, iVal);
              loopInd.emplace_back(accessIndex);
            }
            Value res = createKrnl.load(X, loopInd);
            Value zeroVal = emitConstantOp(rewriter, loc, elementType, 0);
            res = createMath.select(shiftOrSet0, res, zeroVal);
            createKrnl.store(res, bufMemRef, initLoopInd);
          }
        });

    // Outer loop iterates over the number of steps.
    ValueRange stepLoopDef = createKrnl.defineLoops(1);
    createKrnl.iterateIE(stepLoopDef, stepLoopDef, {zero}, {numberOfStep + 1},
        [&](KrnlBuilder &createKrnl, ValueRange stepLoopInd) {
          MathBuilder createMath(createKrnl);

          // Compute index offset: offset = 2^step.
          Value step = stepLoopInd[0];
          step = rewriter.create<IndexCastOp>(loc, i64Ty, step);
          step = rewriter.create<SIToFPOp>(loc, f32Ty, step);
          Value offset = createMath.exp2(step);
          offset = rewriter.create<FPToSIOp>(loc, i64Ty, offset);
          offset = rewriter.create<IndexCastOp>(loc, indexTy, offset);

          // Inner loop iterates over the output to compute sums.
          //   for i range(n):
          //     for k range(m):
          //       if i >= 2^step:
          //         y[i,k] = buf[i - 2^step,k] + buf[i,k]
          //       else:
          //         y[i,k] = buf[i,k]
          ValueRange sumLoopDef = createKrnl.defineLoops(rank);
          createKrnl.iterateIE(sumLoopDef, sumLoopDef, lbs, ubs,
              [&](KrnlBuilder &createKrnl, ValueRange sumLoopInd) {
                IndexExprScope ieScope(createKrnl);
                MathBuilder createMath(createKrnl);
                SymbolIndexExpr axis(axisIE);
                // Load y[i,k].
                Value y1 = createKrnl.load(bufMemRef, sumLoopInd);
                // Load y[i - 2^step,k].
                SmallVector<Value, 4> loopInd;
                Value shouldUpdate = emitConstantOp(rewriter, loc, i1Ty, 0);
                for (uint64_t r = 0; r < rank; ++r) {
                  Value iVal = sumLoopInd[r];
                  Value rVal = emitConstantOp(rewriter, loc, indexTy, r);
                  // Whether we are in the reduction axis.
                  Value isAxis = createMath.eq(rVal, axis.getValue());
                  // Whether the current element's index is in scope.
                  Value isInScope;
                  if (reverse) {
                    Value x = createMath.add(iVal, offset);
                    isInScope = createMath.slt(x, xBounds.getDim(r).getValue());
                  } else {
                    isInScope = createMath.sge(iVal, offset);
                  }
                  Value ok = createMath._and(isAxis, isInScope);
                  shouldUpdate = createMath._or(ok, shouldUpdate);

                  Value iOffset;
                  if (reverse)
                    iOffset = createMath.add(iVal, offset);
                  else
                    iOffset = createMath.sub(iVal, offset);
                  Value accessIndex = createMath.select(ok, iOffset, iVal);
                  loopInd.emplace_back(accessIndex);
                }
                Value y2 = createKrnl.load(bufMemRef, loopInd);
                Value zeroVal = emitConstantOp(rewriter, loc, elementType, 0);
                Value addOrZero = createMath.select(shouldUpdate, y2, zeroVal);
                Value res = createMath.add(y1, addOrZero);
                createKrnl.store(res, resMemRef, sumLoopInd);
              });

          // Reset the temporary buffer to the latest output.
          // buf = y
          ValueRange bufLoopDef = createKrnl.defineLoops(rank);
          createKrnl.iterateIE(bufLoopDef, bufLoopDef, lbs, ubs,
              [&](KrnlBuilder &createKrnl, ValueRange bufLoopInd) {
                Value x = createKrnl.load(resMemRef, bufLoopInd);
                createKrnl.store(x, bufMemRef, bufLoopInd);
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
