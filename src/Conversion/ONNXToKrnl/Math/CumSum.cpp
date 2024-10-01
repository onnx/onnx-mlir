/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- CumSum.cpp - Lowering CumSum Ops ----------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX CumSum Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

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

struct ONNXCumSumOpLowering : public OpConversionPattern<ONNXCumSumOp> {
  ONNXCumSumOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  /// We use a parallel algorithm for cumsum [1] as follows:
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
  /// Blelloch algorithm [2] is more work-efficient. However, it is not
  /// affine-friendly, because the inner bounds depend on the outer bounds.
  ///
  /// [1] Hillis, W. Daniel, and Guy L. Steele, Jr. 1986. "Data Parallel
  /// Algorithms." Communications of the ACM 29(12), pp. 1170–1183.
  ///
  /// [2] Blelloch, Guy E. 1990. "Prefix Sums and Their Applications." Technical
  /// Report CMU-CS-90-190, School of Computer Science, Carnegie Mellon
  /// University.
  LogicalResult matchAndRewrite(ONNXCumSumOp csOp, ONNXCumSumOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = csOp.getOperation();
    Location loc = ONNXLoc<ONNXCumSumOp>(op);

    // Builder helper.
    IndexExprScope mainScope(&rewriter, loc);

    MultiDialectBuilder<KrnlBuilder, MathBuilder, IndexExprBuilderForKrnl,
        MemRefBuilder>
        create(rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

    // Common information.
    Type elementType = memRefType.getElementType();
    Type i64Ty = rewriter.getI64Type();
    Type f32Ty = rewriter.getF32Type();
    Type indexTy = rewriter.getIndexType();

    Value X = adaptor.getX();
    Value axis = adaptor.getAxis();
    bool exclusive = csOp.getExclusive() == 1;
    bool reverse = csOp.getReverse() == 1;

    DimsExpr xDims;
    uint64_t rank = create.krnlIE.getShapedTypeRank(X);
    create.krnlIE.getShapeAsDims(X, xDims);
    LiteralIndexExpr zeroIE(0);

    // Read axis.
    IndexExpr axisIE = create.krnlIE.getIntFromArrayAsSymbol(axis, 0);
    if (axisIE.isUndefined())
      return op->emitError("axis parameter could not be processed");
    axisIE = axisIE.selectOrSelf(axisIE < 0, axisIE + LitIE(rank));

    // Insert an allocation and deallocation for the result of this operation.
    Value resMemRef = create.mem.alignedAlloc(X, memRefType);
    Value bufMemRef = create.mem.alignedAlloc(X, memRefType);

    // Get the size of dimension 'axis'.
    IndexExpr axisSize = LitIE(-1);
    for (uint64_t i = 0; i < rank; ++i)
      axisSize = IndexExpr::select(axisIE == i, xDims[i], axisSize);

    // Compute log2(n), the number of steps.
    IndexExpr numberOfStep;
    if (axisSize.isLiteral()) {
      int64_t n = axisSize.getLiteral();
      int64_t logN = static_cast<int64_t>(std::ceil(std::log2(n)));
      numberOfStep = LitIE(logN);
    } else {
      Value nos = create.math.cast(f32Ty, axisSize.getValue());
      // Use this when math::CeilOp is available in MLIR.
      // nos = create.math.ceil(create.math.log2(nos));
      nos = create.math.log2(nos);
      nos = create.math.cast(i64Ty, nos);
      // Use this when math::CeilOp is available in MLIR.
      // numberOfStep = SymIE(nos);
      numberOfStep = SymIE(nos) + LitIE(1);
    }

    // Input and output have the same shape, so they share the bounds.
    SmallVector<IndexExpr, 4> lbs(rank, zeroIE);
    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(X, ubs);

    // Initialize the temporary buffer: copy values from the input.
    ValueRange initLoopDef = create.krnl.defineLoops(rank);
    create.krnl.iterateIE(initLoopDef, initLoopDef, lbs, ubs,
        [&](const KrnlBuilder &ck, ValueRange initLoopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(ck);
          if (!exclusive) {
            Value x = create.krnl.load(X, initLoopInd);
            create.krnl.store(x, bufMemRef, initLoopInd);
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
            Value axis = axisIE.getValue();

            // Load input[i - 1,k] or get zero.
            SmallVector<Value, 4> loopInd;
            Value offsetOne = create.math.constant(indexTy, 1);
            Value shiftOrSet0 = getLoopIndexByAxisAndOffset(create.math,
                loopInd, initLoopInd, ubs, axis, offsetOne, reverse);
            Value res = create.krnl.load(X, loopInd);
            Value zeroVal = create.math.constant(elementType, 0);
            res = create.math.select(shiftOrSet0, res, zeroVal);
            create.krnl.store(res, bufMemRef, initLoopInd);
          }
        });

    // Outer loop iterates over the number of steps.
    create.krnl.forLoopIE(zeroIE, numberOfStep, /*step*/ 1, /*par*/ false,
        [&](const KrnlBuilder &ck, ValueRange stepLoopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(ck);

          // Compute index offset: offset = 2^step.
          Value step = stepLoopInd[0];
          step = create.math.cast(f32Ty, step);
          Value offset = create.math.exp2(step);
          offset = create.math.castToIndex(offset);

          // Inner loop iterates over the output to compute sums.
          //   for i range(n):
          //     for k range(m):
          //       if i >= 2^step:
          //         y[i,k] = buf[i - 2^step,k] + buf[i,k]
          //       else:
          //         y[i,k] = buf[i,k]
          ValueRange sumLoopDef = create.krnl.defineLoops(rank);
          create.krnl.iterateIE(sumLoopDef, sumLoopDef, lbs, ubs,
              [&](const KrnlBuilder &ck, ValueRange sumLoopInd) {
                IndexExprScope ieScope(ck);
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(ck);
                Value axis = axisIE.getValue();
                // Load buf[i,k].
                Value b1 = create.krnl.load(bufMemRef, sumLoopInd);
                // Load buf[i - 2^step,k].
                SmallVector<Value, 4> loopInd;
                Value shouldUpdate = getLoopIndexByAxisAndOffset(create.math,
                    loopInd, sumLoopInd, ubs, axis, offset, reverse);
                Value b2 = create.krnl.load(bufMemRef, loopInd);
                Value zeroVal = create.math.constant(elementType, 0);
                Value addOrZero = create.math.select(shouldUpdate, b2, zeroVal);
                Value res = create.math.add(b1, addOrZero);
                create.krnl.store(res, resMemRef, sumLoopInd);
              });

          // Reset the temporary buffer to the latest output.
          // buf = y
          ValueRange bufLoopDef = create.krnl.defineLoops(rank);
          create.krnl.iterateIE(bufLoopDef, bufLoopDef, lbs, ubs,
              [&](const KrnlBuilder &createKrnl, ValueRange bufLoopInd) {
                Value x = createKrnl.load(resMemRef, bufLoopInd);
                createKrnl.store(x, bufMemRef, bufLoopInd);
              });
        });

    rewriter.replaceOp(op, resMemRef);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXCumSumOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXCumSumOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
