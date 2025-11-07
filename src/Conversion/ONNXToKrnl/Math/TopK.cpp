/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- TopK.cpp - TopK Op ---------------------------===//
//
// Copyright 2021-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX TopK operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXTopKOpLowering : public OpConversionPattern<ONNXTopKOp> {
  ONNXTopKOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}
  LogicalResult matchAndRewrite(ONNXTopKOp topKOp, ONNXTopKOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = topKOp.getOperation();
    ValueRange operands = adaptor.getOperands();
    Location loc = ONNXLoc<ONNXTopKOp>(op);
    Value X = adaptor.getX();

    // Builders.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Get output memref types.
    Type valuesConvertedType =
        typeConverter->convertType(topKOp.getValues().getType());
    Type indicesConvertedType =
        typeConverter->convertType(topKOp.getIndices().getType());
    assert(valuesConvertedType && mlir::isa<MemRefType>(valuesConvertedType) &&
           "Failed to convert Values type to MemRefType");
    assert(indicesConvertedType &&
           mlir::isa<MemRefType>(indicesConvertedType) &&
           "Failed to convert Indices type to MemRefType");
    MemRefType valuesMemRefType = mlir::cast<MemRefType>(valuesConvertedType);
    MemRefType indicesMemRefType = mlir::cast<MemRefType>(indicesConvertedType);

    // Op's Attributes.
    int64_t rank = valuesMemRefType.getRank();
    int64_t axis = adaptor.getAxis();
    axis = axis < 0 ? axis + rank : axis;
    assert(axis >= 0 && axis < rank && "axis is out of bound");
    bool ascendingMode = adaptor.getLargest() != 1;
    bool sortedMode = topKOp.getSorted() != 0;

    // Get output shape.
    ONNXTopKOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr resDims = shapeHelper.getOutputDims();

    // Get K from the output dimensions (it's already loaded by shapeHelper)
    IndexExpr kIndexExpr = resDims[axis];
    Value kValCasted = kIndexExpr.getValue();

    // Check if we can use the fast path.
    // The fast path in emitTopK requires axis to be the last dim.
    if ((rank > 6) || (axis != (rank - 1))) {
      // TODO: Add support for the generic (slow) path if needed.
      // This would involve re-implementing the bubble sort from emitArgSort
      // or emitting a full sort.
      // For now, we only support the fast path.
      return op->emitError(
          "TopK lowering only supports axis being the last dimension "
          "and rank <= 6");
    }

    // Call the new emitTopK function.
    std::pair<Value, Value> outputs =
        emitTopK(rewriter, loc, X, valuesMemRefType, indicesMemRefType, resDims,
            axis, ascendingMode, kValCasted, sortedMode);

    rewriter.replaceOp(op, {outputs.first, outputs.second});
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXTopKOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXTopKOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
