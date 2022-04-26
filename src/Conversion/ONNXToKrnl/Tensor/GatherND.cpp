/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- GatherND.cpp - Lowering GatherND Op -----------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX GatherND Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gather_nd_onnx_to_krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXGatherNDOpLowering : public ConversionPattern {
  ONNXGatherNDOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXGatherNDOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXGatherNDOpAdaptor operandAdaptor(operands);
    ONNXGatherNDOp gatherNDOp = cast<ONNXGatherNDOp>(op);
    Location loc = op->getLoc();

    ONNXGatherNDOpShapeHelper shapeHelper(&gatherNDOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    // Insert an allocation and deallocation for the result of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Value output = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput());

    // Operands and attributes.
    Value data = operandAdaptor.data();
    Value indices = operandAdaptor.indices();
    int64_t b = gatherNDOp.batch_dims();

    ArrayRef<int64_t> indicesShape =
        indices.getType().cast<ShapedType>().getShape();
    ArrayRef<int64_t> dataShape = data.getType().cast<ShapedType>().getShape();
    int64_t dataRank = dataShape.size();
    int64_t indicesRank = indicesShape.size();
    int64_t outputRank = outputMemRefType.getShape().size();
    int64_t indicesLastDim = indicesShape[indicesRank - 1];

    // Ensure the operation containts are satisfied.
    assert(dataRank >= 1 && "The rank of 'data' must be >= 1");
    assert(indicesRank >= 1 && "The rank of 'indices' must be >= 1");
    assert((outputRank == dataRank + indicesRank - indicesLastDim - 1 - b) &&
           "Incorrect outut rank");
    assert(b >= 0 && "batch_dim should not be negative");
    assert(b < std::min(dataRank, indicesRank) &&
           "batch_dims must be smaller than the min(dataRank, indicesRank)");
    assert((indicesLastDim >= 1 && indicesLastDim <= dataRank - b) &&
           "indices.shape[-1] must be in the range [1, dataRank - b]");

    int64_t batchDimsSize = 1;
    for (int64_t i = 0; i < b; ++i)
      batchDimsSize *= indicesShape[i];

    // Reshape 'indices' to shape [batchDimSize, -1, indices.shape[-1]].
    OnnxToKrnlBuilder create(rewriter, loc);
    LiteralIndexExpr BDS(batchDimsSize), ILD(indicesLastDim);
    LiteralIndexExpr NegOne(-1);
    SmallVector<DimIndexExpr, 6> newIndicesShape = {BDS, NegOne, ILD};
    Value reshapedIndices = create.reshape(indices, newIndicesShape);
    LLVM_DEBUG(llvm::dbgs() << "reshapedIndices: " << reshapedIndices << "\n");

    // Reshape 'data' to shape [batchDimSize, data.shape[b:]
    SmallVector<DimIndexExpr, 6> newDataShape = {BDS};
    for (int64_t i = b; i < dataRank; ++i) {
      LiteralIndexExpr Dim(dataShape[i]);
      newDataShape.emplace_back(Dim);
    }
    Value reshapedData = create.reshape(data, newDataShape);
    LLVM_DEBUG(llvm::dbgs() << "reshapedData: " << reshapedData << "\n");

    //
    //   update_indices = indices.shape[:-1]
    //   for idx in np.ndindex(update_indices):
    //     output[indices[idx]] = updates[idx]
    //

    KrnlBuilder createKrnl(rewriter, loc);
    int64_t reshapedIndicesRank = newIndicesShape.size();
    ValueRange loopDef = createKrnl.defineLoops(reshapedIndicesRank);
    MemRefBoundsIndexCapture reshapedIndicesBounds(reshapedIndices);
    DimsExpr lbs(reshapedIndicesRank, LiteralIndexExpr(0)), ubs;
    reshapedIndicesBounds.getDimList(ubs);

    createKrnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          // Access function for 'reshaped_indices'. Let q = rank(indices).
          // The first (q-1) indexes traverse the iteration space defined by
          // indices.shape[:-1], which corresponds to the first (q-1) induction
          // variable in the loop iteration space.
          DimsExpr reshapedIndicesAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, indicesAccessFct);
          indicesAccessFct.truncate(indicesRank - 1);
        });

    return success();
  }
};

void populateLoweringONNXGatherNDOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXGatherNDOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
