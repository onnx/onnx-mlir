/*
 * SPDX-License-Identifier: Apache-2.0
 */
//===-------SequenceEmpty.cpp - Lowering SequenceEmpty Op-----------------=== //
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX SequenceEmpty Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSequenceEmptyOpLowering
    : public OpConversionPattern<ONNXSequenceEmptyOp> {
  ONNXSequenceEmptyOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXSequenceEmptyOp seqOp,
      ONNXSequenceEmptyOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = seqOp.getOperation();
    Location loc = ONNXLoc<ONNXSequenceEmptyOp>(op);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    Value alloc =
        rewriter.create<KrnlSeqAllocOp>(loc, outputMemRefType, ValueRange());

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXSequenceEmptyOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceEmptyOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
