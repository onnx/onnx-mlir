/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- GatherND.cpp - Lowering GatherND Op -----------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX GatherND Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "gather_nd_onnx_to_krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXGatherNDOpLowering : public OpConversionPattern<ONNXGatherNDOp> {
  ONNXGatherNDOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  // When true causes injection of print stmts in the generated code.
  static constexpr bool emitPrintStmts = false;

  // Debug function used to emit code to print the supplied 'indices'.
  static void printIndices(
      StringRef title, const DimsExpr &indices, const KrnlBuilder &createKrnl) {
    llvm::Twine msg(title + ": (");
    createKrnl.printf(msg.str());
    int64_t n = static_cast<int64_t>(indices.size());
    for (int64_t i = 0; i < n; ++i) {
      Value val = indices[i].getValue();
      createKrnl.printf(" ", val);
    }
    createKrnl.printf(")\n");
  }

  LogicalResult matchAndRewrite(ONNXGatherNDOp gatherNDOp,
      ONNXGatherNDOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = gatherNDOp.getOperation();
    Location loc = ONNXLoc<ONNXGatherNDOp>(op);
    ValueRange operands = adaptor.getOperands();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope outerScope(&rewriter, loc);

    // Get shape.
    ONNXGatherNDOpShapeHelper shapeHelper(
        op, operands, &create.krnlIE, &outerScope);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Operands and attributes.
    Value data = adaptor.getData();
    Value indices = adaptor.getIndices();
    int64_t b = adaptor.getBatchDims();
    DimsExpr dataDims, indicesDims;
    create.krnlIE.getShapeAsDims(data, dataDims);
    create.krnlIE.getShapeAsDims(indices, indicesDims);
    auto dataType = mlir::cast<ShapedType>(data.getType());
    int64_t dataRank = dataDims.size();
    int64_t indicesRank = indicesDims.size();
    auto indicesType = mlir::cast<ShapedType>(indices.getType());
    ArrayRef<int64_t> indicesShape = indicesType.getShape();
    int64_t indicesLastDim = indicesShape[indicesRank - 1];
    // ToFix: Handle case in which indicesLastDim is kDynamic.
    // Currently, such case is detected by ONNXPreKrnlVerifyPass.
    assert((indicesLastDim >= 1 && indicesLastDim <= dataRank - b) &&
           "indices.shape[-1] must be in the range [1, dataRank - b]");

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    DimsExpr outputDims = shapeHelper.getOutputDims();

    // Reshape 'indices' to the 3D shape:
    //   [batchDimSize, indicesDimsSize, indices.shape[-1]].
    LiteralIndexExpr oneIE(1);
    IndexExpr batchDimsSize = oneIE;
    for (int64_t i = 0; i < b; i++)
      batchDimsSize = batchDimsSize * indicesDims[i];
    IndexExpr indicesDimsSize = oneIE;
    for (int64_t i = b; i < indicesRank - 1; i++)
      indicesDimsSize = indicesDimsSize * indicesDims[i];
    IndexExpr BDS(batchDimsSize), IDS(indicesDimsSize);
    LiteralIndexExpr ILD(indicesLastDim);
    DimsExpr newIndicesShape = {BDS, IDS, ILD};
    Value reshapedIndices =
        create.mem.reinterpretCast(indices, newIndicesShape);
    LLVM_DEBUG(llvm::dbgs() << "reshapedIndices: " << reshapedIndices << "\n");

    // Reshape 'data' to shape [batchDimSize, data.shape[b:]]
    DimsExpr newDataDims = {BDS};
    for (int64_t i = b; i < dataRank; ++i) {
      newDataDims.emplace_back(dataDims[i]);
    }
    int64_t reshapedDataRank = newDataDims.size();
    Value reshapedData = create.mem.reinterpretCast(data, newDataDims);
    LLVM_DEBUG(llvm::dbgs() << "reshapedData: " << reshapedData << "\n");

    // Allocate a 1D output buffer.
    IndexExpr outputDimsSize = oneIE;
    for (uint64_t i = 0; i < outputDims.size(); i++)
      outputDimsSize = outputDimsSize * outputDims[i];
    SmallVector<IndexExpr> outputIndexExpr = {outputDimsSize};
    int64_t dim = outputDimsSize.isLiteral() ? outputDimsSize.getLiteral()
                                             : ShapedType::kDynamic;
    Type outputType = dataType.getElementType();
    Value outputDataBuffer =
        create.mem.alloc(MemRefType::get({dim}, outputType), outputIndexExpr);
    // Initialize the index used to store the result values.
    Value iZero = create.math.constantIndex(0);
    Value iOne = create.math.constantIndex(1);
    // Scalar, ok to use alloca.
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
    DimsExpr lbs(2, LitIE(0)), ubs = {newIndicesShape[0], newIndicesShape[1]};

    if (emitPrintStmts) {
      create.krnl.printTensor("reshapedIndices%s%d%e", reshapedIndices);
      create.krnl.printTensor("reshapedData%s%d%e", reshapedData);
    }

    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          // Access function for 'reshapedIndices'. The first 2 indices are
          // equal to the loop indexes.
          DimsExpr reshapedIndicesAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, reshapedIndicesAccessFct);

          // Access function for 'reshapedData'. The first index is equal to the
          // first loop index.
          DimsExpr reshapedDataAccessFct;
          IndexExpr ind = SymIE(loopInd[0]);
          reshapedDataAccessFct.emplace_back(ind);

          // The last index of the access function for 'reshapedIndices' is
          // given by the values of indices.shape[-1].
          // The loaded values from 'reshapedIndices' are the next set of
          // indices to push to the `reshapedDataAccessFct`.
          for (unsigned i = 0; i < indicesLastDim; ++i) {
            IndexExpr ind = LitIE(i);
            reshapedIndicesAccessFct.emplace_back(ind);

            if (emitPrintStmts)
              printIndices("indices", reshapedIndicesAccessFct, createKrnl);

            Value indexVal =
                createKrnl.loadIE(reshapedIndices, reshapedIndicesAccessFct);
            reshapedIndicesAccessFct.pop_back();

            if (emitPrintStmts) {
              createKrnl.printf("index = ", indexVal);
              createKrnl.printf("\n");
            }

            IndexExpr index = NonAffineIndexExpr(indexVal);
            reshapedDataAccessFct.emplace_back(index);
          }

          if (indicesLastDim == dataRank - b) {
            // When indices.shape[-1] is equal to (rank(data) - b) the
            // `reshapedDataAccessFct` computed so far has the same number of
            // indices as the rank of 'reshapedData'.
            assert(static_cast<int64_t>(reshapedDataAccessFct.size()) ==
                       reshapedDataRank &&
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
            Value zero = create.math.constantIndex(0);
            IndexExpr reshapedDataLastDimExpr = dataDims[dataRank - 1];
            Value last = reshapedDataLastDimExpr.getValue();
            ValueRange innerLoopDef = create.krnl.defineLoops(1);
            create.krnl.iterate(innerLoopDef, innerLoopDef, {zero}, {last},
                [&](const KrnlBuilder &createKrnl, ValueRange innerLoopInd) {
                  IndexExpr ind = SymIE(innerLoopInd[0]);
                  reshapedDataAccessFct.emplace_back(ind);
                  assert(static_cast<int64_t>(reshapedDataAccessFct.size()) ==
                             reshapedDataRank &&
                         "Access function should have the same rank as "
                         "reshapedData");

                  if (emitPrintStmts)
                    printIndices(
                        "data indices", reshapedDataAccessFct, createKrnl);

                  // Gather value from the 'data' tensor and store it into
                  // 'outputDataBuffer'.
                  Value val =
                      createKrnl.loadIE(reshapedData, reshapedDataAccessFct);
                  reshapedDataAccessFct.pop_back();

                  if (emitPrintStmts) {
                    createKrnl.printf("val = ", val);
                    createKrnl.printf("\n");
                  }

                  Value storeIndexVal = createKrnl.load(storeIndex);
                  createKrnl.store(val, outputDataBuffer, storeIndexVal);

                  // Bump up the storeIndex.
                  createKrnl.store(
                      create.math.add(storeIndexVal, iOne), storeIndex);
                });
          }
        });

    // Finally reshape 'outputDataBuffer' to the shape of the output.
    Value reshapedOutput =
        create.mem.reinterpretCast(outputDataBuffer, outputDims);
    LLVM_DEBUG(llvm::dbgs() << "reshapedOutput: " << reshapedOutput << "\n");

    rewriter.replaceOp(op, reshapedOutput);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXGatherNDOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXGatherNDOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
