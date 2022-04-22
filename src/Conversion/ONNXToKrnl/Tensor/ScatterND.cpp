/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- ScatterND.cpp - Lowering ScatterND Op ----------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ScatterND Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXScatterNDOpLowering : public ConversionPattern {
  ONNXScatterNDOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXScatterNDOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXScatterNDOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Operands and attributes.
    Value data = operandAdaptor.data();
    Value updates = operandAdaptor.updates();
    Value indices = operandAdaptor.indices();
    auto dataType = data.getType().cast<ShapedType>();
    auto indicesType = indices.getType().cast<ShapedType>();
    auto updatesType = updates.getType().cast<ShapedType>();
    int64_t dataRank = dataType.getRank();
    int64_t updatesRank = updatesType.getRank();
    int64_t indicesRank = indicesType.getRank();

    assert(dataRank >= 1 && "The rank of 'data' must be >= 1");
    assert(indicesRank >= 1 && "The rank of 'indices' must be >= 1");

    // Insert an allocation and deallocation for the result of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    int64_t outputRank = outputMemRefType.getShape().size();
    assert(outputRank == dataRank && "Output rank not equal to data rank");

    KrnlBuilder createKrnl(rewriter, loc);
    IndexExprScope indexScope(createKrnl);
    MemRefBoundsIndexCapture dataBounds(data);
    DimsExpr dataDims;
    dataBounds.getDimList(dataDims);
    Value output = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, dataDims);

    // Step1: copy `data` into `output`.
    Value sizeInBytes = getDynamicMemRefSizeInBytes(rewriter, loc, data);
    createKrnl.memcpy(output, data, sizeInBytes);

    // Step2: scatter the updates values into the output.
    //   update_indices = indices.shape[:-1]
    //   for idx in np.ndindex(update_indices):
    //     output[indices[idx]] = updates[idx]
    //
    ValueRange loopDef = createKrnl.defineLoops(updatesRank);
    DimsExpr lbs(updatesRank, LiteralIndexExpr(0)), ubs;
    MemRefBoundsIndexCapture updatesBounds(updates);
    updatesBounds.getDimList(ubs);

    createKrnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          // Access function for 'indices'. Let q = rank(indices).
          // The first (q-1) indexes traverse the iteration space defined by
          // indices.shape[:-1], which corresponds to the first (q-1) induction
          // variable in the loop iteration space.
          DimsExpr indicesAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, indicesAccessFct);
          indicesAccessFct.truncate(indicesRank - 1);

          // Access function for the output. Let r=rank(data), q=rank(indices).
          // The first indices.shape[-1] indexes are given by looking up the
          // 'indices' tensor. The remaining (r-q-1) indexes are given by the
          // loop iteration space.
          DimsExpr outputAccessFct;
          for (unsigned i = 0; i < dataRank; ++i) {
            if (i < indicesRank - 1) {
              IndexExpr ind = LiteralIndexExpr(i);
              indicesAccessFct.emplace_back(ind);
              Value indexVal = createKrnl.loadIE(indices, indicesAccessFct);
              IndexExpr index = NonAffineIndexExpr(indexVal);
              outputAccessFct.emplace_back(index);
            } else {
              IndexExpr index = SymbolIndexExpr(loopInd[i]);
              outputAccessFct.emplace_back(index);
            }
          }

          // Scatter 'update' values into the output tensor.
          Value updateVal = createKrnl.load(updates, loopInd);
          createKrnl.storeIE(updateVal, output, outputAccessFct);
        });

    rewriter.replaceOp(op, output);

    return success();
  }
};

void populateLoweringONNXScatterNDOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXScatterNDOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
