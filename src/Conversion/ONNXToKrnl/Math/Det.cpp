/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Det.cpp - Lowering Det Op --------------------------===//
//
// This file lowers the ONNX Det Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXDetOpLowering : public OpConversionPattern<ONNXDetOp> {
  bool enableParallel = false;

  ONNXDetOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern<ONNXDetOp>(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXDetOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ONNXDetOp detOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = detOp.getOperation();
    Location loc = ONNXLoc<ONNXDetOp>(op);
    ValueRange operands = adaptor.getOperands();

    IndexExprScope scope(&rewriter, loc);
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    ONNXDetOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr outputDims = shapeHelper.getOutputDims();

    Value X = adaptor.getX();
    MemRefType xType = mlir::cast<MemRefType>(X.getType());
    Type elementType = xType.getElementType();
    int64_t xRank = xType.getRank();
    int64_t batchRank = xRank - 2;

    Type convertedType =
        this->typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Allocate the (batched) output.
    Value alloc = create.mem.alignedAlloc(outputMemRefType, outputDims);

    // The innermost two dimensions form a square matrix (see the shape helper).
    Value M = create.krnlIE.getShapeAsDim(X, xRank - 2).getValue();
    Value oneVal = create.math.constant(elementType, 1.0);
    Value negOneVal = create.math.constant(elementType, -1.0);
    Value zeroVal = create.math.constant(elementType, 0.0);

    MemRefType scratchType = MemRefType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, elementType);

    // Batch dimensions are independent of each other, so they may be
    // computed in parallel. There is nothing to parallelize when the input
    // is plain 2-D (single, scalar-output "batch").
    bool doParallel = enableParallel && batchRank > 0;
    if (enableParallel && batchRank == 0)
      onnxToKrnlParallelReport(
          op, false, -1, 0, "no batch dim to parallelize for Det");
    else if (enableParallel)
      onnxToKrnlParallelReport(
          op, true, 0, LitIE(0), outputDims[0], "Det batch loop");

    SmallVector<IndexExpr, 4> batchLbs(batchRank, LitIE(0));
    SmallVector<int64_t, 4> batchSteps(batchRank, 1);
    SmallVector<bool, 4> batchParallel(batchRank, doParallel);

    create.krnl.forLoopsIE(batchLbs, outputDims, batchSteps, batchParallel,
        [&](const KrnlBuilder &createKrnl, ValueRange batchIndices) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
              createKrnl);
          SmallVector<IndexExpr, 2> zeros(2, LitIE(0));
          SmallVector<IndexExpr, 2> mMs(2, SymIE(M));
          SmallVector<int64_t, 2> ones(2, 1);
          SmallVector<bool, 2> noPar(2, false);

          // Scratch copy of the current batch's MxM matrix. The elimination
          // mutates it in place; the running determinant and the pivot search
          // are carried as krnl.iterate loop results instead.
          Value scratch =
              create.mem.alignedAlloc(scratchType, ValueRange{M, M});

          // Copy X[batchIndices, :, :] into the scratch matrix.
          create.krnl.forLoopsIE(zeros, mMs, ones, noPar,
              [&](const KrnlBuilder &createKrnl, ValueRange ij) {
                SmallVector<Value, 4> xIndices(
                    batchIndices.begin(), batchIndices.end());
                xIndices.append(ij.begin(), ij.end());
                Value v = createKrnl.load(X, xIndices);
                createKrnl.store(v, scratch, ij);
              });

          // Gaussian elimination with partial pivoting. The running
          // determinant (product of pivots, sign-flipped on every row swap)
          // is the sole loop-carried value of the k-loop.
          ValueRange kLoop = create.krnl.defineLoops(1);
          auto kIterate = create.krnl.iterateIE(kLoop, kLoop, {LitIE(0)},
              {SymIE(M)}, ValueRange{oneVal},
              [&](const KrnlBuilder &createKrnl, ValueRange kIndices,
                  ValueRange kIterArgs) {
                IndexExprScope kScope(createKrnl);
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);
                Value k = kIndices[0];
                Value detIn = kIterArgs[0];

                // Find the pivot row: index in [k, M) with the largest
                // absolute value in column k, carried as (bestVal, bestRow).
                Value initAbs =
                    create.math.abs(create.krnl.load(scratch, {k, k}));
                ValueRange rLoop = create.krnl.defineLoops(1);
                auto pivotIterate = create.krnl.iterateIE(rLoop, rLoop,
                    {DimIE(k) + 1}, {SymIE(M)}, ValueRange{initAbs, k},
                    [&](const KrnlBuilder &createKrnl, ValueRange rIndices,
                        ValueRange pivotArgs) {
                      MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                          createKrnl);
                      Value r = rIndices[0];
                      Value absVal =
                          create.math.abs(create.krnl.load(scratch, {r, k}));
                      Value isBetter = create.math.sgt(absVal, pivotArgs[0]);
                      Value newBestVal =
                          create.math.select(isBetter, absVal, pivotArgs[0]);
                      Value newBestRow =
                          create.math.select(isBetter, r, pivotArgs[1]);
                      create.krnl.yield({newBestVal, newBestRow});
                    });
                Value pivotRow = pivotIterate.getResult(1);

                // Swap rows k and pivotRow (a no-op when they coincide) and
                // flip the sign of the running determinant if they differ.
                create.krnl.forLoopsIE({LitIE(0)}, {SymIE(M)}, {1}, {false},
                    [&](const KrnlBuilder &createKrnl, ValueRange cIndices) {
                      Value c = cIndices[0];
                      Value a = createKrnl.load(scratch, {k, c});
                      Value b = createKrnl.load(scratch, {pivotRow, c});
                      createKrnl.store(b, scratch, {k, c});
                      createKrnl.store(a, scratch, {pivotRow, c});
                    });
                Value rowsDiffer = create.math.neq(pivotRow, k);
                Value sign = create.math.select(rowsDiffer, negOneVal, oneVal);
                Value detAfterSwap = create.math.mul(detIn, sign);

                // Multiply the running determinant by the (post-swap) pivot.
                Value pivotElem = create.krnl.load(scratch, {k, k});
                Value detAfterPivot = create.math.mul(detAfterSwap, pivotElem);

                // Eliminate the entries below the pivot. When the pivot is
                // zero the matrix is singular and the determinant is already
                // zero, so the elimination factor is forced to zero instead
                // of dividing by the pivot.
                Value pivotNonZero = create.math.neq(pivotElem, zeroVal);
                create.krnl.forLoopsIE({DimIE(k) + 1}, {SymIE(M)}, {1}, {false},
                    [&](const KrnlBuilder &createKrnl, ValueRange iIndices) {
                      IndexExprScope iScope(createKrnl);
                      MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                          createKrnl);
                      Value i = iIndices[0];
                      Value ik = create.krnl.load(scratch, {i, k});
                      Value ratio = create.math.div(ik, pivotElem);
                      Value factor =
                          create.math.select(pivotNonZero, ratio, zeroVal);
                      create.krnl.forLoopsIE({DimIE(k)}, {SymIE(M)}, {1},
                          {false},
                          [&](const KrnlBuilder &createKrnl,
                              ValueRange jIndices) {
                            MultiDialectBuilder<KrnlBuilder, MathBuilder>
                                create(createKrnl);
                            Value j = jIndices[0];
                            Value kj = create.krnl.load(scratch, {k, j});
                            Value ij = create.krnl.load(scratch, {i, j});
                            Value newVal = create.math.sub(
                                ij, create.math.mul(factor, kj));
                            create.krnl.store(newVal, scratch, {i, j});
                          });
                    });
                create.krnl.yield({detAfterPivot});
              });

          // Store the accumulated determinant into the output.
          create.krnl.store(kIterate.getResult(0), alloc, batchIndices);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXDetOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXDetOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
