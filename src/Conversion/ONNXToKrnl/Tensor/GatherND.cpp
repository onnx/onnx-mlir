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
#include <numeric>

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

    // Operands and attributes.
    Value data = operandAdaptor.data();
    Value indices = operandAdaptor.indices();
    int64_t b = gatherNDOp.batch_dims();

    auto indicesType = indices.getType().cast<ShapedType>();
    auto dataType = data.getType().cast<ShapedType>();
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    ArrayRef<int64_t> indicesShape = indicesType.getShape();
    ArrayRef<int64_t> dataShape = dataType.getShape();
    ArrayRef<int64_t> outputShape = outputMemRefType.getShape();
    int64_t dataRank = dataShape.size();
    int64_t indicesRank = indicesShape.size();
    int64_t outputRank = outputShape.size();
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

    // Reshape 'indices' to the 3D shape:
    //   [batchDimSize, indicesDimsSize, indices.shape[-1]].
    OnnxToKrnlBuilder create(rewriter, loc);
    int64_t batchDimsSize = std::accumulate(indicesShape.begin(),
        indicesShape.begin() + b, 1, std::multiplies<int64_t>());
    int64_t indicesDimsSize = std::accumulate(indicesShape.begin(),
        indicesShape.end(), 1, std::multiplies<int64_t>());
    LiteralIndexExpr BDS(batchDimsSize),
        IDS(indicesDimsSize / (batchDimsSize * indicesLastDim)),
        ILD(indicesShape[indicesRank - 1]);
    DimsExpr newIndicesShape = {BDS, IDS, ILD};
    Value reshapedIndices =
        emitMemRefReinterpretCastOp(rewriter, loc, indices, newIndicesShape);
    LLVM_DEBUG(llvm::dbgs() << "reshapedIndices: " << reshapedIndices << "\n");

    // Reshape 'data' to shape [batchDimSize, data.shape[b:]]
    DimsExpr newDataShape = {BDS};
    for (int64_t i = b; i < dataRank; ++i) {
      LiteralIndexExpr dataDim(dataShape[i]);
      newDataShape.emplace_back(dataDim);
    }
    Value reshapedData =
        emitMemRefReinterpretCastOp(rewriter, loc, data, newDataShape);
    LLVM_DEBUG(llvm::dbgs() << "reshapedData: " << reshapedData << "\n");

    // for (i,j) in (0..reshapedIndices.shape[0]), 0..reshapedIndices.shape[1])
    // {
    //   idx = tuple(reshapedIndices[i][j])
    //   output.append(reshapedData[(i,) + idx])
    // }
    // output.reshape(outputShape)
    KrnlBuilder createKrnl(rewriter, loc);
    ValueRange loopDef = createKrnl.defineLoops(2);
    MemRefBoundsIndexCapture reshapedIndicesBounds(reshapedIndices);
    DimsExpr lbs(2, LiteralIndexExpr(0)), ubs;
    reshapedIndicesBounds.getDimList(ubs);
    ubs.truncate(2);

    MemRefBuilder createMemRef(rewriter, loc);
    Value outputDataBuffer = createMemRef.alloc(MemRefType::get(
        {reshapedIndicesBounds.getShape(0) * reshapedIndicesBounds.getShape(1)},
        outputMemRefType.getElementType()));

    createKrnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          // Access function for 'reshapedIndices'. The first 2 indices are
          // simply the the loop indexes.
          DimsExpr reshapedIndicesAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, reshapedIndicesAccessFct);

          // Access function for 'reshapedData'. The first index is given by
          // the batch_dims value.
          DimsExpr reshapedDataAccessFct;
          IndexExpr ind = LiteralIndexExpr(b);
          reshapedDataAccessFct.emplace_back(ind);

          // The last index of the access function for 'reshapedIndices' is
          // given by the value of of indices.shape[-1].
          // The loaded values from 'reshapedIndices' are the indices for
          // 'reshapedData'.
          for (unsigned i = 0; i < indicesShape[indicesRank - 1]; ++i) {
            IndexExpr ind = LiteralIndexExpr(i);
            reshapedIndicesAccessFct.emplace_back(ind);
            Value indexVal =
                createKrnl.loadIE(reshapedIndices, reshapedIndicesAccessFct);
            reshapedIndicesAccessFct.pop_back();
            IndexExpr index = NonAffineIndexExpr(indexVal);
            reshapedDataAccessFct.emplace_back(index);
          }

          // Gather values from the 'data' tensor and save them.
          Value val = createKrnl.loadIE(reshapedData, reshapedDataAccessFct);
          IndexExpr storeIE =
              SymbolIndexExpr(loopInd[0]) *
                  LiteralIndexExpr(reshapedIndicesBounds.getShape(1)) +
              SymbolIndexExpr(loopInd[1]);
          createKrnl.storeIE(val, outputDataBuffer, storeIE);
        });

    DimsExpr newOutputShape;
    for (int64_t dim : outputShape) {
      LiteralIndexExpr outputDim(dim);
      newOutputShape.emplace_back(outputDim);
    }

    Value reshapedOutput = emitMemRefReinterpretCastOp(
        rewriter, loc, outputDataBuffer, newOutputShape);
    LLVM_DEBUG(llvm::dbgs() << "reshapedOutput: " << reshapedOutput << "\n");

    rewriter.replaceOp(op, reshapedOutput);

    return success();
  }
};

void populateLoweringONNXGatherNDOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXGatherNDOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
