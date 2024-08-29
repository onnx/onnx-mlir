/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- Dim.cpp - Lowering Dim Op ----------------------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Dim Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXDimOpLowering : public OpConversionPattern<ONNXDimOp> {
  ONNXDimOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXDimOp dimOp, ONNXDimOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Get basic info.
    Operation *op = dimOp.getOperation();
    Location loc = ONNXLoc<ONNXDimOp>(op);
    Value data = adaptor.getData();
    int64_t axis = adaptor.getAxis();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(&rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    Type elementType = outputMemRefType.getElementType();

    // Output is 1D memref of one element.
    SmallVector<IndexExpr, 1> outputDims(1, LitIE(1));
    Value alloc = create.mem.alignedAlloc(outputMemRefType, outputDims);

    // Write the dimension at axis to the output.
    Value dimValue = create.krnlIE.getShapeAsDim(data, axis).getValue();
    dimValue = create.math.cast(elementType, dimValue);
    Value index = create.math.constantIndex(0);
    create.krnl.store(dimValue, alloc, {index});

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXDimOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXDimOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
