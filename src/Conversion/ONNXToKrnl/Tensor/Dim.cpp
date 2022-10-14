/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- Dim.cpp - Lowering Dim Op ----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Dim Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXDimOpLowering : public ConversionPattern {
  ONNXDimOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXDimOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get basic info.
    Location loc = op->getLoc();
    auto dimOp = llvm::dyn_cast<ONNXDimOp>(op);
    ONNXDimOpAdaptor operandAdaptor(operands);
    Value data = operandAdaptor.data();
    int64_t axis = dimOp.axis();
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
    IndexExprScope scope(&rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    Type elementType = outputMemRefType.getElementType();

    // Output is 1D memref of one element.
    SmallVector<IndexExpr, 1> outputDims(1, LiteralIndexExpr(1));
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, outputDims);

    // Write the dimension at axis to the output.
    MemRefBoundsIndexCapture dataBounds(data);
    Value dimValue = dataBounds.getDim(axis).getValue();
    dimValue = create.math.cast(elementType, dimValue);
    Value index = create.math.constantIndex(0);
    create.krnl.store(dimValue, alloc, {index});

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXDimOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXDimOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
