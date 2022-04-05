/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- CumSum.cpp - Lowering CumSum Ops ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX CumSum Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

static Value getLoopIndexByAxisAndOffset(MathBuilder &createMath,
    SmallVectorImpl<Value> &resLoopIndex, ValueRange &baseLoopIndex,
    SmallVectorImpl<IndexExpr> &upperBounds, Value axis, Value offset,
    bool reverse) {
  Type boolTy = createMath.getBuilder().getI1Type();
  Type indexTy = createMath.getBuilder().getIndexType();
  Value notSameAsBaseIndex = createMath.constant(boolTy, 0);
  for (uint64_t r = 0; r < upperBounds.size(); ++r) {
    Value iVal = baseLoopIndex[r];
    Value rVal = createMath.constant(indexTy, r);
    Value dimSize = upperBounds[r].getValue();

    // Whether we are in the right axis.
    Value isAxis = createMath.eq(rVal, axis);

    // Whether (index - offset) (or index + offset in case of reverse) is still
    // in the valid range or not.
    Value iOffset, isValidOffset;
    if (reverse) {
      iOffset = createMath.add(iVal, offset);
      isValidOffset = createMath.slt(iOffset, dimSize);
    } else {
      Value zero = createMath.constant(indexTy, 0);
      iOffset = createMath.sub(iVal, offset);
      isValidOffset = createMath.sge(iOffset, zero);
    }

    Value ok = createMath.andi(isAxis, isValidOffset);
    notSameAsBaseIndex = createMath.ori(ok, notSameAsBaseIndex);

    Value accessIndex = createMath.select(ok, iOffset, iVal);
    resLoopIndex.emplace_back(accessIndex);
  }
  return notSameAsBaseIndex;
}

struct ONNXCumSumOpLowering : public ConversionPattern {
  ONNXCumSumOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXCumSumOp::getOperationName(), 1, ctx) {}

  /// We use a paralel algorithm for cumsum [1] as follows:
  /// Assume that input is x whose shape in [n,m], and axis for cumsum is 0.
  /// We double-buffer the output to avoid intermediate result being overwritten
  /// by multiple threads.
  /// ```
  /// buf = x
  /// for step in range(log2(n)):
  ///   for i in range(n):
  ///     for k in range(m):
  ///       if i >= 2^step:
  ///         y[i,k] = buf[i - 2^step,k] + buf[i,k]
  ///       else:
  ///         y[i,k] = buf[i,k]
  ///   buf = y
  /// ```
  ///
  /// Blelloch algorithm [2] is more work-efficent. However, it is not
  /// affine-friendly, because the inner bounds depend on the outer bounds.
  ///
  /// [1] Hillis, W. Daniel, and Guy L. Steele, Jr. 1986. "Data Parallel
  /// Algorithms." Communications of the ACM 29(12), pp. 1170–1183.
  ///
  /// [2] Blelloch, Guy E. 1990. "Prefix Sums and Their Applications." Technical
  /// Report CMU-CS-90-190, School of Computer Science, Carnegie Mellon
  /// University.
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXCumSumOp csOp = llvm::cast<ONNXCumSumOp>(op);
    ONNXCumSumOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Builder helper.
    IndexExprScope mainScope(&rewriter, loc);
    KrnlBuilder createKrnl(rewriter, loc);
    MathBuilder createMath(createKrnl);

    // Common information.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = memRefType.getElementType();
    Type i64Ty = rewriter.getI64Type();
    Type f32Ty = rewriter.getF32Type();
    Type indexTy = rewriter.getIndexType();

    Value X = operandAdaptor.x();
    Value axis = operandAdaptor.axis();
    bool exclusive = csOp.exclusive() == 1;
    bool reverse = csOp.reverse() == 1;

    MemRefBoundsIndexCapture xBounds(X);
    uint64_t rank = xBounds.getRank();
    LiteralIndexExpr zeroIE(0);

    // Read axis.
    ArrayValueIndexCapture axisCapture(axis,
        getDenseElementAttributeFromConstantValue,
        krnl::loadDenseElementArrayValueAtIndex);
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
      int64_t logn = (int64_t)std::ceil(std::log2(n));
      numberOfStep = LiteralIndexExpr(logn);
    } else {
      Value nos = createMath.cast(f32Ty, axisSize.getValue());
      // Use this when math::CeilOp is available in MLIR.
      // nos = createMath.ceil(createMath.log2(nos));
      nos = createMath.log2(nos);
      nos = createMath.cast(i64Ty, nos);
      // Use this when math::CeilOp is available in MLIR.
      // numberOfStep = SymbolIndexExpr(nos);
      numberOfStep = SymbolIndexExpr(nos) + LiteralIndexExpr(1);
    }

    // Input and output have the same shape, so they share the bounds.
    SmallVector<IndexExpr, 4> lbs(rank, zeroIE);
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
            // Exclusive mode is equivalent to shifting all elements right (left
            // if reversed) and set the first element (the last element if
            // reversed) to 0.
            //
            // For example, doing exclusive mode on the input:
            //   input = [2, 3, 4]
            // is equivalent to doing non-exclusive mode on:
            //   new_input = [0, 2, 3]
            // or
            //   new_input = [3, 4, 0] if reversed.
            MathBuilder createMath(createKrnl);
            Value axis = axisIE.getValue();

            // Load input[i - 1,k] or get zero.
            SmallVector<Value, 4> loopInd;
            Value offsetOne = createMath.constant(indexTy, 1);
            Value shiftOrSet0 = getLoopIndexByAxisAndOffset(createMath, loopInd,
                initLoopInd, ubs, axis, offsetOne, reverse);
            Value res = createKrnl.load(X, loopInd);
            Value zeroVal = createMath.constant(elementType, 0);
            res = createMath.select(shiftOrSet0, res, zeroVal);
            createKrnl.store(res, bufMemRef, initLoopInd);
          }
        });

    // Outer loop iterates over the number of steps.
    ValueRange stepLoopDef = createKrnl.defineLoops(1);
    createKrnl.iterateIE(stepLoopDef, stepLoopDef, {zeroIE}, {numberOfStep},
        [&](KrnlBuilder &createKrnl, ValueRange stepLoopInd) {
          MathBuilder createMath(createKrnl);

          // Compute index offset: offset = 2^step.
          Value step = stepLoopInd[0];
          step = createMath.cast(f32Ty, step);
          Value offset = createMath.exp2(step);
          offset = createMath.castToIndex(offset);

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
                Value axis = axisIE.getValue();
                // Load buf[i,k].
                Value b1 = createKrnl.load(bufMemRef, sumLoopInd);
                // Load buf[i - 2^step,k].
                SmallVector<Value, 4> loopInd;
                Value shouldUpdate = getLoopIndexByAxisAndOffset(createMath,
                    loopInd, sumLoopInd, ubs, axis, offset, reverse);
                Value b2 = createKrnl.load(bufMemRef, loopInd);
                Value zeroVal = createMath.constant(elementType, 0);
                Value addOrZero = createMath.select(shouldUpdate, b2, zeroVal);
                Value res = createMath.add(b1, addOrZero);
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

void populateLoweringONNXCumSumOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXCumSumOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
