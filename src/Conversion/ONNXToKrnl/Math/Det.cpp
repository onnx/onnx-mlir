/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- Det.cpp - Lowering Det Op
//------------------------===//
//
// This file lowers the ONNX Det Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

// Lowers Det by computing the determinant of each MxM submatrix (the two
// innermost dims) via Gaussian elimination with partial pivoting on a
// scratch copy, one batch element at a time. Independent batch elements are
// optionally computed in parallel. Within a batch, the elimination has
// loop-carried state (running product, pivot row/value, in-place row
// updates); the running product and the pivot search are threaded as
// scf.for/scf.if results (not memory), so that only the scratch matrix
// itself (which is genuinely mutated in place) needs a backing buffer. That
// buffer is intentionally never explicitly deallocated here: memref.dealloc
// is illegal in this pass (see ConvertONNXToKrnl.cpp), freeing is handled by
// the later buffer-deallocation pass, same as every other Krnl lowering.
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
        MemRefBuilder, SCFBuilder>
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

    // Matrix dimension (M), and useful constants.
    Value M = create.krnlIE.getShapeAsDim(X, xRank - 1).getValue();
    Value zero = create.math.constantIndex(0);
    Value one = create.math.constantIndex(1);
    Value zeroVal = create.math.constant(elementType, 0.0);
    Value oneVal = create.math.constant(elementType, 1.0);
    Value negOneVal = create.math.constant(elementType, -1.0);

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

    create.scf.forLoopsIE(batchLbs, outputDims, batchSteps, batchParallel,
        [&](const SCFBuilder &createSCF, ValueRange batchIndices) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder,
              SCFBuilder>
              create(createSCF);

          // Scratch copy of the current batch's MxM matrix. This is the
          // only buffer that genuinely needs backing storage, since the
          // elimination mutates it in place; everything else (running
          // determinant, pivot search) is carried as SSA loop results.
          Value scratch =
              create.mem.alignedAlloc(scratchType, ValueRange{M, M});

          // Copy X[batchIndices, :, :] into the scratch matrix.
          create.scf.forLoop(
              zero, M, 1, [&](const SCFBuilder &createSCF, ValueRange iv) {
                MultiDialectBuilder<KrnlBuilder, SCFBuilder> create(createSCF);
                Value i = iv[0];
                create.scf.forLoop(zero, M, 1,
                    [&](const SCFBuilder &createSCF, ValueRange jv) {
                      KrnlBuilder createKrnl(createSCF);
                      Value j = jv[0];
                      SmallVector<Value, 4> xIndices(
                          batchIndices.begin(), batchIndices.end());
                      xIndices.push_back(i);
                      xIndices.push_back(j);
                      Value v = createKrnl.load(X, xIndices);
                      createKrnl.store(v, scratch, ValueRange{i, j});
                    });
              });

          // Gaussian elimination with partial pivoting. The running
          // determinant (product of pivots, sign-flipped on every row
          // swap) is the sole loop-carried value of the k-loop.
          scf::ForOp kForOp = scf::ForOp::create(create.krnl.getBuilder(),
              create.krnl.getLoc(), zero, M, one, ValueRange{oneVal},
              [&](OpBuilder &kBuilder, Location kLoc, Value k,
                  ValueRange kIterArgs) {
                Value detIn = kIterArgs[0];
                MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder>
                    create(kBuilder, kLoc);

                // Find the pivot row: index in [k, M) with the largest
                // absolute value in column k. Carried as (bestVal, bestRow).
                Value initVal = create.krnl.load(scratch, ValueRange{k, k});
                Value initAbs = create.math.abs(initVal);
                Value kPlus1 = create.math.add(k, one);

                scf::ForOp pivotForOp = scf::ForOp::create(kBuilder, kLoc,
                    kPlus1, M, one, ValueRange{initAbs, k},
                    [&](OpBuilder &rBuilder, Location rLoc, Value r,
                        ValueRange pivotArgs) {
                      MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                          rBuilder, rLoc);
                      Value bestVal = pivotArgs[0];
                      Value bestRow = pivotArgs[1];
                      Value val = create.krnl.load(scratch, ValueRange{r, k});
                      Value absVal = create.math.abs(val);
                      Value isBetter = create.math.sgt(absVal, bestVal);
                      Value newBestVal =
                          create.math.select(isBetter, absVal, bestVal);
                      Value newBestRow =
                          create.math.select(isBetter, r, bestRow);
                      scf::YieldOp::create(
                          rBuilder, rLoc, ValueRange{newBestVal, newBestRow});
                    });
                Value pivotRow = pivotForOp.getResult(1);

                // Swap rows k and pivotRow if they differ, flipping the
                // sign of the running determinant. Built manually (rather
                // than via the then/else-callback convenience overload)
                // since that overload does not accept an explicit
                // TypeRange in this MLIR version and always produces a
                // zero-result op, which would mismatch the single value
                // yielded below.
                Value rowsDiffer = create.math.neq(pivotRow, k);
                auto swapIfOp = scf::IfOp::create(kBuilder, kLoc,
                    TypeRange{elementType}, rowsDiffer, /*addThenBlock=*/true,
                    /*addElseBlock=*/true);
                {
                  OpBuilder::InsertionGuard guard(kBuilder);
                  kBuilder.setInsertionPointToStart(
                      &swapIfOp.getThenRegion().front());
                  MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder>
                      create(kBuilder, kLoc);
                  create.scf.forLoop(zero, M, 1,
                      [&](const SCFBuilder &createSCF, ValueRange cv) {
                        KrnlBuilder createKrnl(createSCF);
                        Value c = cv[0];
                        Value a = createKrnl.load(scratch, ValueRange{k, c});
                        Value b =
                            createKrnl.load(scratch, ValueRange{pivotRow, c});
                        createKrnl.store(b, scratch, ValueRange{k, c});
                        createKrnl.store(a, scratch, ValueRange{pivotRow, c});
                      });
                  Value negDet = create.math.mul(detIn, negOneVal);
                  scf::YieldOp::create(kBuilder, kLoc, ValueRange{negDet});
                }
                {
                  OpBuilder::InsertionGuard guard(kBuilder);
                  kBuilder.setInsertionPointToStart(
                      &swapIfOp.getElseRegion().front());
                  scf::YieldOp::create(kBuilder, kLoc, ValueRange{detIn});
                }
                Value detAfterSwap = swapIfOp.getResult(0);

                // Multiply the running determinant by the (post-swap)
                // pivot.
                Value pivotElem = create.krnl.load(scratch, ValueRange{k, k});
                Value detAfterPivot = create.math.mul(detAfterSwap, pivotElem);

                // Eliminate the entries below the pivot. Guarded: when the
                // pivot is zero the matrix is singular, the determinant is
                // already zero, and dividing by the pivot must be avoided.
                Value pivotNonZero = create.math.neq(pivotElem, zeroVal);
                create.scf.ifThenElse(
                    pivotNonZero, [&](const SCFBuilder &createSCF) {
                      MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder>
                          create(createSCF);
                      Value kPlus1b = create.math.add(k, one);
                      create.scf.forLoop(kPlus1b, M, 1,
                          [&](const SCFBuilder &createSCF, ValueRange iv) {
                            MultiDialectBuilder<KrnlBuilder, MathBuilder,
                                SCFBuilder>
                                create(createSCF);
                            Value i = iv[0];
                            Value ik =
                                create.krnl.load(scratch, ValueRange{i, k});
                            Value factor = create.math.div(ik, pivotElem);
                            create.scf.forLoop(k, M, 1,
                                [&](const SCFBuilder &createSCF,
                                    ValueRange jv) {
                                  MultiDialectBuilder<KrnlBuilder, MathBuilder>
                                      create(createSCF);
                                  Value j = jv[0];
                                  Value kj = create.krnl.load(
                                      scratch, ValueRange{k, j});
                                  Value ijVal = create.krnl.load(
                                      scratch, ValueRange{i, j});
                                  Value newVal = create.math.sub(
                                      ijVal, create.math.mul(factor, kj));
                                  create.krnl.store(
                                      newVal, scratch, ValueRange{i, j});
                                });
                          });
                    });

                scf::YieldOp::create(kBuilder, kLoc, ValueRange{detAfterPivot});
              });
          Value finalDet = kForOp.getResult(0);

          // Store the accumulated determinant into the output.
          create.krnl.store(finalDet, alloc, batchIndices);
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
