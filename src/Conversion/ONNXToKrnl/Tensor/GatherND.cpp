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

  // When true causes injection of print stmts in the generated code.
  static constexpr bool emitPrintStmts = false;

  // Debug function used to emit code to print the supplied 'indices'.
  static void printIndices(
      StringRef title, const DimsExpr &indices, KrnlBuilder &createKrnl) {
    llvm::Twine msg(title + ": (");
    createKrnl.printf(msg.str());
    int64_t n = (int64_t)indices.size();
    for (int64_t i = 0; i < n; ++i) {
      Value val = indices[i].getValue();
      createKrnl.printf(val, val.getType());
    }
    createKrnl.printf(")\n");
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXGatherNDOpAdaptor operandAdaptor(operands);
    ONNXGatherNDOp gatherNDOp = cast<ONNXGatherNDOp>(op);
    Location loc = op->getLoc();
    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);
    IndexExprScope outerScope(&rewriter, loc);

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
    ArrayRef<int64_t> indicesShape = indicesType.getShape();
    ArrayRef<int64_t> dataShape = dataType.getShape();
    int64_t dataRank = dataShape.size();
    int64_t indicesRank = indicesShape.size();
    int64_t indicesLastDim = indicesShape[indicesRank - 1];

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    ArrayRef<int64_t> outputShape = outputMemRefType.getShape();
    int64_t outputRank = outputShape.size();

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
    const int64_t batchDimsSize = std::accumulate(indicesShape.begin(),
        indicesShape.begin() + b, 1, std::multiplies<int64_t>());
    const int64_t indicesDimsSize = std::accumulate(indicesShape.begin(),
        indicesShape.end(), 1, std::multiplies<int64_t>());
    assert(batchDimsSize >= 0 && "batchDimsSize must be non-negative");
    assert(indicesDimsSize >= 0 && "indicesDimsSize must be non-negative");

    LiteralIndexExpr BDS(batchDimsSize),
        IDS(indicesDimsSize / (batchDimsSize * indicesLastDim)),
        ILD(indicesLastDim);
    DimsExpr newIndicesShape = {BDS, IDS, ILD};
    Value reshapedIndices =
        create.mem.reinterpretCast(indices, newIndicesShape);
    LLVM_DEBUG(llvm::dbgs() << "reshapedIndices: " << reshapedIndices << "\n");

    // Reshape 'data' to shape [batchDimSize, data.shape[b:]]
    DimsExpr newDataShape = {BDS};
    for (int64_t i = b; i < dataRank; ++i) {
      assert(dataShape[i] > 0 && "Cannot support data with dynamic dimensions");
      LiteralIndexExpr dataDim(dataShape[i]);
      newDataShape.emplace_back(dataDim);
    }
    int64_t reshapedDataRank = newDataShape.size();
    Value reshapedData = create.mem.reinterpretCast(data, newDataShape);
    LLVM_DEBUG(llvm::dbgs() << "reshapedData: " << reshapedData << "\n");

    // Allocate a 1D output buffer.
    const int64_t outputDimsSize = std::accumulate(
        outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
    Value outputDataBuffer = create.mem.alloc(
        MemRefType::get({outputDimsSize}, outputMemRefType.getElementType()));

    // Initialize the index used to store the result values.
    Value iZero = create.math.constantIndex(0);
    Value iOne = create.math.constantIndex(1);
    Value storeIndex =
        create.mem.alloca(MemRefType::get({}, rewriter.getIndexType()));
    create.krnl.store(iZero, storeIndex);

    // for (i,j) in (0..reshapedIndices.shape[0]), 0..reshapedIndices.shape[1])
    // {
    //   idx = tuple(reshapedIndices[i][j])
    //   output.append(reshapedData[(i,) + idx])
    // }
    // output.reshape(outputShape)
    ValueRange loopDef = create.krnl.defineLoops(2);
    DimsExpr lbs(2, LiteralIndexExpr(0)),
        ubs = {newIndicesShape[0], newIndicesShape[1]};

    if (emitPrintStmts) {
      create.krnl.printTensor("reshapedIndices: ", reshapedIndices);
      create.krnl.printTensor("reshapedData: ", reshapedData);
    }

    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          // Access function for 'reshapedIndices'. The first 2 indices are
          // equal to the loop indexes.
          DimsExpr reshapedIndicesAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, reshapedIndicesAccessFct);

          // Access function for 'reshapedData'. The first index is equal to the
          // first loop index.
          DimsExpr reshapedDataAccessFct;
          IndexExpr ind = SymbolIndexExpr(loopInd[0]);
          reshapedDataAccessFct.emplace_back(ind);

          // The last index of the access function for 'reshapedIndices' is
          // given by the values of indices.shape[-1].
          // The loaded values from 'reshapedIndices' are the next set of
          // indices to push to the `reshapedDataAccessFct`.
          for (unsigned i = 0; i < indicesLastDim; ++i) {
            IndexExpr ind = LiteralIndexExpr(i);
            reshapedIndicesAccessFct.emplace_back(ind);

            if (emitPrintStmts)
              printIndices("indices", reshapedIndicesAccessFct, createKrnl);

            Value indexVal =
                createKrnl.loadIE(reshapedIndices, reshapedIndicesAccessFct);
            reshapedIndicesAccessFct.pop_back();

            if (emitPrintStmts) {
              createKrnl.printf("index = ", indexVal, indexVal.getType());
              createKrnl.printf("\n");
            }

            IndexExpr index = NonAffineIndexExpr(indexVal);
            reshapedDataAccessFct.emplace_back(index);
          }

          if (indicesLastDim == dataRank - b) {
            // When indices.shape[-1] is equal to (rank(data) - b) the
            // `reshapedDataAccessFct` computed so far has the same number of
            // indices as the rank of 'reshapedData'.
            assert((int64_t)reshapedDataAccessFct.size() == reshapedDataRank &&
                   "Access function should have the same rank as reshapedData");

            if (emitPrintStmts)
              printIndices("data indices", reshapedDataAccessFct, createKrnl);

            // Gather value from the 'data' tensor and store it into
            // 'outputDataBuffer'.
            Value val = createKrnl.loadIE(reshapedData, reshapedDataAccessFct);
            Value storeIndexVal = createKrnl.load(storeIndex);
            createKrnl.store(val, outputDataBuffer, storeIndexVal);

            // Bump up the storeIndex.
            createKrnl.store(create.math.add(storeIndexVal, iOne), storeIndex);
          } else {
            assert((indicesLastDim < dataRank - b) &&
                   "Expecting indices.shape[-1] to be smaller than "
                   "rank(indices) - b");

            // When indices.shape[-1] is less than (rank(data) - b) the
            // `reshapedDataAccessFct` computed so far yields a slice which
            // needs to be inserted into the output buffer.
            int64_t reshapedDataLastDim = dataShape[dataRank - 1];
            for (int64_t i = 0; i < reshapedDataLastDim; ++i) {
              IndexExpr ind = LiteralIndexExpr(i);
              reshapedDataAccessFct.emplace_back(ind);
              assert(
                  (int64_t)reshapedDataAccessFct.size() == reshapedDataRank &&
                  "Access function should have the same rank as reshapedData");

              if (emitPrintStmts)
                printIndices("data indices", reshapedDataAccessFct, createKrnl);

              // Gather value from the 'data' tensor and store it into
              // 'outputDataBuffer'.
              Value val =
                  createKrnl.loadIE(reshapedData, reshapedDataAccessFct);
              reshapedDataAccessFct.pop_back();

              if (emitPrintStmts) {
                createKrnl.printf("val = ", val, val.getType());
                createKrnl.printf("\n");
              }

              Value storeIndexVal = createKrnl.load(storeIndex);
              createKrnl.store(val, outputDataBuffer, storeIndexVal);

              // Bump up the storeIndex.
              createKrnl.store(
                  create.math.add(storeIndexVal, iOne), storeIndex);
            }
          }
        });

    // Finally reshape 'outputDataBuffer' to the shape of the output.
    DimsExpr newOutputShape;
    for (int64_t dim : outputShape) {
      LiteralIndexExpr outputDim(dim);
      newOutputShape.emplace_back(outputDim);
    }

    Value reshapedOutput =
        create.mem.reinterpretCast(outputDataBuffer, newOutputShape);
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
