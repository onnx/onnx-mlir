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

struct ONNXScatterNDOpLowering : public ConversionPattern {
  ONNXScatterNDOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXScatterNDOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXScatterNDOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Operands and attributes.
    Value data = operandAdaptor.getData();
    Value updates = operandAdaptor.getUpdates();
    Value indices = operandAdaptor.getIndices();
    auto dataType = data.getType().cast<ShapedType>();
    auto indicesType = indices.getType().cast<ShapedType>();
    auto updatesType = updates.getType().cast<ShapedType>();
    int64_t dataRank = dataType.getRank();
    int64_t updatesRank = updatesType.getRank();
    int64_t indicesRank = indicesType.getRank();

    assert(dataRank >= 1 && "The rank of 'data' must be >= 1");
    assert(indicesRank >= 1 && "The rank of 'indices' must be >= 1");

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
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
