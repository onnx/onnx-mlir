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
  ONNXTensorScatterOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

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
    //MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Insert an allocation and deallocation for the result of this operation.
    IndexExprScope indexScope(create.krnl);
    DimsExpr dataDims;
    create.krnlIE.getShapeAsDims(pastCache, dataDims);

    // Step1: the output reuse the buffer of pastCache
    Value output = pastCache;

    // Runtime check for out of bound can be added
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
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
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
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXTensorScatterOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
