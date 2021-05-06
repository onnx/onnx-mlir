/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Gather.cpp - Lowering Gather Op----------------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gather Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXGatherOpLowering : public ConversionPattern {
  ONNXGatherOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXGatherOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXGatherOpAdaptor operandAdaptor(operands);
    ONNXGatherOp gatherOp = llvm::cast<ONNXGatherOp>(op);
    auto loc = op->getLoc();

    ONNXGatherOpShapeHelper shapeHelper(&gatherOp, rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));
    // Scope for krnl EDSC ops
    using namespace mlir::edsc;
    ScopedContext scope(rewriter, loc);
    IndexExprScope outerScope(shapeHelper.scope);

    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = outputMemRefType.getElementType();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Save axis and rank info.
    int64_t axisLit = gatherOp.axis();
    int64_t dataRank = shapeHelper.dataDims.size();
    int64_t indicesRank = shapeHelper.indicesDims.size();
    int64_t outputRank = shapeHelper.dimsForOutput(0).size();

    /*
      The pattern that we are using is that of numpy.take.

      Ni, Nk = data.shape[:axis], data.shape[axis+1:]
      Nj = indices.shape
      for ii in ndindex(Ni):
        for jj in ndindex(Nj):
          for kk in ndindex(Nk):
            out[ii + jj + kk] = data[ii + (indices[jj],) + kk]
    */
    // Define loops and iteration trip counts (equivalent to size of output)
    BuildKrnlLoop outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
    outputLoops.createIterateOp();
    int iIndexStart = 0;
    int jIndexStart = iIndexStart + axisLit;
    int kIndexStart = jIndexStart + indicesRank - (axisLit + 1);

    // Insert code inside the loop.
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());
    IndexExprScope innerLoopScope;
    LiteralIndexExpr zero(0);
    LiteralIndexExpr axis(axisLit);
    SymbolIndexExpr axisDim(shapeHelper.dataDims[axisLit]);

    // compute the loop indices for the output
    SmallVector<IndexExpr, 4> outputAccessFct;
    getIndexExprList<DimIndexExpr>(
        outputLoops.getAllInductionVar(), outputAccessFct);

    // Compute access function for indices[jjs].
    SmallVector<IndexExpr, 4> indicesAccessFct;
    for (int j = 0; j < indicesRank; ++j)
      indicesAccessFct.emplace_back(outputAccessFct[jIndexStart + j]);
    Value indexVal = krnl_load(operandAdaptor.indices(), indicesAccessFct);
    // Loaded value is an index that is not affine
    IndexExpr index = NonAffineIndexExpr(indexVal);
    // When index may be negative, add axis Dim to it.
    if (!shapeHelper.positiveConstantIndices) {
      index = index.selectOrSelf(index < zero, index + axisDim);
    }

    // Compute access function of data: data[ii + (indices[jj],) + kk]
    SmallVector<IndexExpr, 4> dataAccessFct;
    // First add indices iis
    for (int i = 0; i < axisLit; ++i)
      dataAccessFct.emplace_back(outputAccessFct[iIndexStart + i]);
    // Then add indices[jj] (indexVal).
    dataAccessFct.emplace_back(index);
    // Then add kks.
    for (int k = axisLit + 1; k < dataRank; ++k)
      dataAccessFct.emplace_back(outputAccessFct[kIndexStart + k]);
    Value data = krnl_load(operandAdaptor.data(), dataAccessFct);

    // Save data into output
    krnl_store(data, alloc, outputAccessFct);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXGatherOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXGatherOpLowering>(ctx);
}
