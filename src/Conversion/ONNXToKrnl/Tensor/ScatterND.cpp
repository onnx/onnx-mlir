/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- ScatterND.cpp - Lowering ScatterND Op ----------------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ScatterND Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXScatterNDOpLowering : public OpConversionPattern<ONNXScatterNDOp> {
  ONNXScatterNDOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXScatterNDOp scatterNDOp,
      ONNXScatterNDOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = scatterNDOp.getOperation();
    Location loc = ONNXLoc<ONNXScatterNDOp>(op);

    // Operands and attributes.
    Value data = adaptor.getData();
    Value updates = adaptor.getUpdates();
    Value indices = adaptor.getIndices();
    auto dataType = mlir::cast<ShapedType>(data.getType());
    auto indicesType = mlir::cast<ShapedType>(indices.getType());
    auto updatesType = mlir::cast<ShapedType>(updates.getType());
    int64_t dataRank = dataType.getRank();
    int64_t updatesRank = updatesType.getRank();
    int64_t indicesRank = indicesType.getRank();

    assert(dataRank >= 1 && "The rank of 'data' must be >= 1");
    assert(indicesRank >= 1 && "The rank of 'indices' must be >= 1");

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t outputRank = outputMemRefType.getShape().size();
    assert(outputRank == dataRank && "Output rank not equal to data rank");

    // Insert an allocation and deallocation for the result of this operation.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope indexScope(create.krnl);
    DimsExpr dataDims;
    create.krnlIE.getShapeAsDims(data, dataDims);
    Value output = create.mem.alignedAlloc(outputMemRefType, dataDims);

    // Step1: copy `data` into `output`.
    Value numOfElements = getDynamicMemRefSize(rewriter, loc, data);
    create.krnl.memcpy(output, data, numOfElements);

    // Step2: scatter the updates values into the output.
    //   update_indices = indices.shape[:-1]
    //   for idx in np.ndindex(update_indices):
    //     output[indices[idx]] = updates[idx]
    //
    ValueRange loopDef = create.krnl.defineLoops(updatesRank);
    DimsExpr lbs(updatesRank, LiteralIndexExpr(0)), ubs;
    create.krnlIE.getShapeAsDims(updates, ubs);
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          // Access function for 'indices'. Let q = rank(indices).
          // The first (q-1) indexes traverse the iteration space defined by
          // indices.shape[:-1], which corresponds to the first (q-1) induction
          // variable in the loop iteration space.
          DimsExpr indicesAccessFctFirst;
          getIndexExprList<DimIndexExpr>(loopInd, indicesAccessFctFirst);
          indicesAccessFctFirst.truncate(indicesRank - 1);

          // Access function for the output. Let r=rank(data), q=rank(indices).
          // The first indices.shape[-1] indexes are given by looking up the
          // 'indices' tensor. The remaining (r-q-1) indexes are given by the
          // loop iteration space.
          DimsExpr outputAccessFct;
          for (unsigned i = 0; i < dataRank; ++i) {
            if (i < indicesRank - 1) {
              IndexExpr ind = LiteralIndexExpr(i);
              DimsExpr indicesAccessFct(indicesAccessFctFirst);
              indicesAccessFct.emplace_back(ind);
              Value indexVal = createKrnl.loadIE(indices, indicesAccessFct);
              IndexExpr index = NonAffineIndexExpr(indexVal);
              outputAccessFct.emplace_back(index);
            } else {
              IndexExpr index = SymbolIndexExpr(
                  loopInd[std::min<unsigned>(i, loopInd.size() - 1)]);
              outputAccessFct.emplace_back(index);
            }
          }

          // Scatter 'update' values into the output tensor.
          Value updateVal = createKrnl.load(updates, loopInd);
          createKrnl.storeIE(updateVal, output, outputAccessFct);
        });

    rewriter.replaceOp(op, output);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXScatterNDOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXScatterNDOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
