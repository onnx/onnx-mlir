/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------SequenceAt.cpp - Lowering SequenceAt Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX SequenceAt Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXSequenceAtOpLowering : public ConversionPattern {
  ONNXSequenceAtOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSequenceAtOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);

    ONNXSequenceAtOpAdaptor operandAdaptor(operands);

    auto input_sequence = operandAdaptor.input_sequence();
    MemRefBoundsIndexCapture inputBounds(input_sequence);
    IndexExpr positionIE = SymbolIndexExpr(operandAdaptor.position());
    positionIE = IndexExpr::select(positionIE < 0, positionIE + inputBounds.getDim(0), positionIE);

    auto outputVal = create.krnl.load(operandAdaptor.input_sequence(), positionIE.getValue());

    rewriter.replaceOp(op, outputVal);
    return success();
  }
};

void populateLoweringONNXSequenceAtOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceAtOpLowering>(typeConverter, ctx);
}
