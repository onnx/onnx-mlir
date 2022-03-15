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
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXSequenceLengthOpLowering : public ConversionPattern {
  ONNXSequenceLengthOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXSequenceLengthOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXSequenceLengthOpAdaptor operandAdaptor(operands);
    MultiDialectBuilder<MemRefBuilder> create(rewriter, loc);
    IndexExprScope IEScope(&rewriter, loc);

    auto input = operandAdaptor.input_sequence();
    auto outputVal = create.mem.dim(input, 0);

    rewriter.replaceOp(op, outputVal);
    return success();
  }
};

void populateLoweringONNXSequenceLengthOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceLengthOpLowering>(typeConverter, ctx);
}
