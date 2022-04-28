/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- ScatterElements.cpp - Lowering ScatterElements Op ----------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ScatterElements Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXScatterElementsOpLowering : public ConversionPattern {
  ONNXScatterElementsOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXScatterElementsOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXScatterElementsOpAdaptor operandAdaptor(operands);
    ONNXScatterElementsOp scatterElements = cast<ONNXScatterElementsOp>(op);
    Location loc = op->getLoc();

    // Operands and attributes.
    Value data = operandAdaptor.data();
    Value updates = operandAdaptor.updates();
    Value indices = operandAdaptor.indices();
    int64_t axis = scatterElements.axis();
    int64_t dataRank = data.getType().cast<MemRefType>().getRank();
    int64_t updatesRank = updates.getType().cast<MemRefType>().getRank();
    int64_t indicesRank = indices.getType().cast<MemRefType>().getRank();
    assert(updatesRank == dataRank && indicesRank == dataRank &&
           "All input tenstors must have the same rank");

    // Determine whether indices may be negative.
    bool indicesMayBeNegative = !indicesAreNonNegativeConstants(indices);

    // Negative value means counting dimensions from the back.
    axis = axis < 0 ? axis + dataRank : axis;

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    int64_t outputRank = outputMemRefType.getShape().size();
    assert(outputRank == dataRank && "Output rank not equal to data rank");

    // Insert an allocation and deallocation for the result of this operation.
    KrnlBuilder createKrnl(rewriter, loc);
    IndexExprScope indexScope(createKrnl);
    MemRefBoundsIndexCapture dataBounds(data);
    DimsExpr dataDims;
    dataBounds.getDimList(dataDims);
    Value output = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, dataDims);

    // Step1: copy the data array into the output array.
    Value sizeInBytes = getDynamicMemRefSizeInBytes(rewriter, loc, data);
    createKrnl.memcpy(output, data, sizeInBytes);

    // Step2: scatter the updates array into the output array.
    //   index = indices[i][j]...[n]
    //   val = updates[i][j]...[n]
    //   output[i][j]..[index]..[n] = val (index used at position axis)
    //
    ValueRange loopDef = createKrnl.defineLoops(updatesRank);
    DimsExpr lbs(updatesRank, LiteralIndexExpr(0)), ubs;
    MemRefBoundsIndexCapture updatesBounds(updates);
    updatesBounds.getDimList(ubs);
    createKrnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          // Access function for updates and indices.
          SmallVector<IndexExpr, 4> accessFct;
          getIndexExprList<DimIndexExpr>(loopInd, accessFct);

          Value updateVal = createKrnl.loadIE(updates, accessFct);
          Value indexVal = createKrnl.loadIE(indices, accessFct);
          IndexExpr index = NonAffineIndexExpr(indexVal);

          // When index may be negative, add axis dim to it.
          if (indicesMayBeNegative) {
            LiteralIndexExpr zero(0);
            SymbolIndexExpr axisDim(dataDims[axis]);
            index = index.selectOrSelf(index < zero, index + axisDim);
          }

          // Access function for the output.
          SmallVector<IndexExpr, 4> outputAccessFct;
          for (int i = 0; i < dataRank; ++i)
            outputAccessFct.emplace_back((i == axis) ? index : accessFct[i]);

          // Scatter updateVal into the output tensor.
          createKrnl.storeIE(updateVal, output, outputAccessFct);
        });

    rewriter.replaceOp(op, output);

    return success();
  }
};

void populateLoweringONNXScatterElementsOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXScatterElementsOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
