/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Gather.cpp - Lowering Gather Op ---------------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gather Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXGatherOpLowering : public ConversionPattern {
  ONNXGatherOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXGatherOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXGatherOpAdaptor operandAdaptor(operands);
    ONNXGatherOp gatherOp = cast<ONNXGatherOp>(op);
    Location loc = op->getLoc();

    ONNXGatherOpShapeHelper shapeHelper(&gatherOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput());

    // Operands and attributes.
    Value data = operandAdaptor.data();
    Value indices = operandAdaptor.indices();
    int64_t axisLit = gatherOp.axis();
    int64_t dataRank = data.getType().cast<MemRefType>().getRank();
    int64_t indicesRank = indices.getType().cast<MemRefType>().getRank();

    // Determine whether indices may be negative.
    bool indicesMayBeNegative = !indicesAreNonNegativeConstants(indices);

    // Negative value means counting dimensions from the back.
    axisLit = axisLit < 0 ? axisLit + dataRank : axisLit;

    int64_t outputRank = shapeHelper.dimsForOutput().size();
    int iIndexStart = 0;
    int jIndexStart = iIndexStart + axisLit;
    int kIndexStart = jIndexStart + indicesRank - (axisLit + 1);

    LiteralIndexExpr zeroIE(0);
    MemRefBoundsIndexCapture dataBounds(data);
    DimsExpr dataDims;
    dataBounds.getDimList(dataDims);

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
    KrnlBuilder createKrnl(rewriter, loc);
    ValueRange loopDef = createKrnl.defineLoops(outputRank);
    DimsExpr lbs(outputRank, zeroIE);
    createKrnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.dimsForOutput(),
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);
          SymbolIndexExpr axisDim(dataDims[axisLit]);

          // compute the loop indices for the output
          SmallVector<IndexExpr, 4> outputAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, outputAccessFct);

          // Compute access function for indices[jjs].
          SmallVector<IndexExpr, 4> indicesAccessFct;
          for (int j = 0; j < indicesRank; ++j)
            indicesAccessFct.emplace_back(outputAccessFct[jIndexStart + j]);
          Value indexVal = createKrnl.loadIE(indices, indicesAccessFct);
          // Loaded value is an index that is not affine
          IndexExpr index = NonAffineIndexExpr(indexVal);
          // When index may be negative, add axis Dim to it.
          if (indicesMayBeNegative)
            index = index.selectOrSelf(index < zeroIE, index + axisDim);

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
          Value dataVal = createKrnl.loadIE(data, dataAccessFct);

          // Save data into output
          createKrnl.storeIE(dataVal, alloc, outputAccessFct);
        });
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXGatherOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXGatherOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
