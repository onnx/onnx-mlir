/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- GatherElements.cpp - Lowering GatherElements Op ----------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX GatherElements Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXGatherElementsOpLowering : public ConversionPattern {
  ONNXGatherElementsOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXGatherElementsOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXGatherElementsOpAdaptor operandAdaptor(operands);
    ONNXGatherElementsOp gatherElementsOp = cast<ONNXGatherElementsOp>(op);
    Location loc = op->getLoc();

    ONNXGatherElementsOpShapeHelper shapeHelper(&gatherElementsOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();

    // Insert an allocation and deallocation for the result of this operation.
    Value output = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput());

    // Operands and attributes.
    Value data = operandAdaptor.data();
    Value indices = operandAdaptor.indices();
    int64_t axis = gatherElementsOp.axis();
    int64_t dataRank = data.getType().cast<MemRefType>().getRank();
    int64_t indicesRank = indices.getType().cast<MemRefType>().getRank();
    int64_t outputRank = outputMemRefType.getShape().size();
    assert(indicesRank == dataRank && "Input tensors must have the same rank");
    assert(outputRank == dataRank && "Output rank not equal to data rank");

    // Determine whether indices may be negative.
    bool indicesMayBeNegative = !indicesAreNonNegativeConstants(indices);

    // Negative value means counting dimensions from the back.
    axis = axis < 0 ? axis + dataRank : axis;

    MemRefBoundsIndexCapture dataBounds(data), indicesBounds(indices);
    DimsExpr dataDims, indicesDims;
    dataBounds.getDimList(dataDims);
    indicesBounds.getDimList(indicesDims);

    // Gather elements from the 'data' tensor, store them into the output.
    //   index = indices[i][j]...[n]
    //   output[i][j]...[n] = data[i][j]..[index]..[n] (index used at axis dim.)
    //
    KrnlBuilder createKrnl(rewriter, loc);
    ValueRange loopDef = createKrnl.defineLoops(indicesRank);
    DimsExpr lbs(indicesRank, LiteralIndexExpr(0));
    createKrnl.iterateIE(loopDef, loopDef, lbs, indicesDims,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          // Access function for indices and output.
          DimsExpr accessFct;
          getIndexExprList<DimIndexExpr>(loopInd, accessFct);

          // Compute index = indices[i][j]...[n]
          Value indexVal = createKrnl.loadIE(indices, accessFct);
          IndexExpr index = NonAffineIndexExpr(indexVal);

          if (indicesMayBeNegative) {
            LiteralIndexExpr zero(0);
            SymbolIndexExpr axisDim(dataDims[axis]);
            index = index.selectOrSelf(index < zero, index + axisDim);
          }

          // Access function for the 'data' tensor.
          DimsExpr dataAccessFct;
          for (int64_t i = 0; i < dataRank; ++i)
            dataAccessFct.emplace_back((i == axis) ? index : accessFct[i]);

          // Gather values from the 'data' tensor and save them.
          Value dataVal = createKrnl.loadIE(data, dataAccessFct);
          createKrnl.storeIE(dataVal, output, accessFct);
        });

    rewriter.replaceOp(op, output);

    return success();
  }
};

void populateLoweringONNXGatherElementsOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXGatherElementsOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
