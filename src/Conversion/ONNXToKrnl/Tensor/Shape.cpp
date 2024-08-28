/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Shape.cpp - Lowering Shape Op ----------------------===//
//
// Copyright 2020-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Shape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXShapeOpLowering : public OpConversionPattern<ONNXShapeOp> {
  ONNXShapeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXShapeOp shapeOp, ONNXShapeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = shapeOp.getOperation();
    Location loc = ONNXLoc<ONNXShapeOp>(op);
    ValueRange operands = adaptor.getOperands();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXShapeOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    Type elementType = outputMemRefType.getElementType();

    // TODO: if the dimensions are known at compile time
    // (shapeHelper.dimsForOutput literal), then we could use a constant array.
    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Compute the data selected by the Shape operator.
    DimsExpr selectedData;
    shapeHelper.computeSelectedDataShape(selectedData);

    // Iterate along the data shape storing dim value to result.
    for (uint64_t i = 0; i < selectedData.size(); ++i) {
      Value val = selectedData[i].getValue();
      Value intVal = create.math.cast(elementType, val);
      create.krnl.storeIE(intVal, alloc, {LitIE(i)});
    }
    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXShapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXShapeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
