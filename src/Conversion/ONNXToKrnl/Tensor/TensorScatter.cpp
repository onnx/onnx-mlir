/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- TensorScatter.cpp - Lowering TensorScatter Op --------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX TensorScatter Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXTensorScatterOpLowering
    : public OpConversionPattern<ONNXTensorScatterOp> {
  ONNXTensorScatterOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXTensorScatterOp::getOperationName());
  }

private:
  bool enableParallel = false;

public:
  LogicalResult matchAndRewrite(ONNXTensorScatterOp tensorScatterOp,
      ONNXTensorScatterOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = tensorScatterOp.getOperation();
    Location loc = ONNXLoc<ONNXTensorScatterOp>(op);

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Operands and attributes.
    Value pastCache = adaptor.getPastCache();
    Value update = adaptor.getUpdate();
    Value writeIndices = adaptor.getWriteIndices();
    bool hasWriteIndices = !isNoneValue(writeIndices);
    int64_t axis = adaptor.getAxis();
    bool isCircular = adaptor.getMode() == "circular";

    int64_t rank = mlir::cast<MemRefType>(pastCache.getType()).getRank();
    assert(mlir::cast<MemRefType>(update.getType()).getRank() == rank &&
           "'past_cache' and 'update' must have the same rank");

    // Negative value means counting dimensions from the back.
    axis = axis < 0 ? axis + rank : axis;

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");

    // Insert an allocation and deallocation for the result of this operation.
    IndexExprScope indexScope(create.krnl);
    DimsExpr dataDims;
    create.krnlIE.getShapeAsDims(pastCache, dataDims);

    // Step1: the output reuse the buffer of pastCache
    Value output = pastCache;

    // Step2: scatter the 'update' values into the output.
    //   for idx in np.ndindex(update.shape):
    //     batch_idx = idx[0]
    //     cache_idx = idx, but with idx[axis] replaced by
    //         write_indices[batch_idx] + idx[axis]   (or just idx[axis] when
    //         'write_indices' is not provided), wrapped modulo the cache's
    //         'axis' dimension when 'mode' is "circular".
    //     output[cache_idx] = update[idx]
    ValueRange loopDef = create.krnl.defineLoops(rank);
    DimsExpr lbs(rank, LitIE(0)), ubs;
    create.krnlIE.getShapeAsDims(update, ubs);

    // Enable parallelism if required. Every iteration writes a distinct
    // output element (the access function is injective: for a fixed batch,
    // the write position along 'axis' is a strictly increasing function of
    // the loop index at 'axis', and every other dimension is copied as-is),
    // and only reads from 'update' and 'write_indices', so the loop nest is
    // safe to parallelize over any subset of its dimensions.
    // bodyCost 1: one innermost iteration is a load, some index arithmetic,
    // and a store.
    KrnlParallelPlan plan = KrnlParallelPlan::noCollapse(loopDef,
        /*parFirstInclusiveDim=*/0, /*parLastExclusiveDim=*/2,
        {.minTripCountForParallel = 4, .bodyCost = 1});
    if (enableParallel)
      plan.tryCreateParallel(create.krnl, op, "tensor scatter", lbs, ubs);

    create.krnl.iterateIE(loopDef, plan.optimizedLoopDef(), lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          DimsExpr accessFct;
          getIndexExprList<DimIndexExpr>(loopInd, accessFct);

          // Compute the write position along 'axis'.
          IndexExpr axisIndex = accessFct[axis];
          if (hasWriteIndices) {
            IndexExpr batchIdx = accessFct[0];
            Value offsetVal = createKrnl.loadIE(writeIndices, {batchIdx});
            IndexExpr offset = NonAffineIndexExpr(offsetVal);
            axisIndex = offset + axisIndex;
          }
          if (isCircular) {
            SymbolIndexExpr axisDim(dataDims[axis]);
            axisIndex = axisIndex % axisDim;
          }

          // Access function for the output: same as 'update's access
          // function, except along 'axis'.
          DimsExpr outputAccessFct(accessFct);
          outputAccessFct[axis] = axisIndex;

          Value updateVal = createKrnl.loadIE(update, accessFct);
          createKrnl.storeIE(updateVal, output, outputAccessFct);
        });

    rewriter.replaceOp(op, output);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXTensorScatterOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXTensorScatterOpLowering>(
      typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
