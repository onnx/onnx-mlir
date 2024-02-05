/*
 * SPDX-License-Identifier: Apache-2.0
 */
//===---------SequenceLength.cpp - Lowering SequenceLength Op-------------=== //
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX SequenceLength Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSequenceLengthOpLowering
    : public OpConversionPattern<ONNXSequenceLengthOp> {
  ONNXSequenceLengthOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXSequenceLengthOp seqOp,
      ONNXSequenceLengthOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = seqOp.getOperation();
    Location loc = ONNXLoc<ONNXSequenceLengthOp>(op);

    MultiDialectBuilder<MemRefBuilder> create(rewriter, loc);
    IndexExprScope IEScope(&rewriter, loc);

    Value input = adaptor.getInputSequence();
    Value outputVal = create.mem.dim(input, 0);

    rewriter.replaceOp(op, outputVal);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXSequenceLengthOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceLengthOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
