/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------SequenceAt.cpp - Lowering SequenceAt-----------------=== //
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
      : ConversionPattern(typeConverter,
            mlir::ONNXSequenceAtOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXSequenceAtOpAdaptor operandAdaptor(operands);
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
    IndexExprScope IEScope(&rewriter, loc);

    auto input_sequence = operandAdaptor.input_sequence();
    auto dimSize = rewriter.create<memref::DimOp>(
        loc, input_sequence, create.math.constantIndex(0));
    SymbolIndexExpr boundIE(dimSize);
    IndexExpr positionIE =
        SymbolIndexExpr(create.krnl.load(operandAdaptor.position()));

    positionIE =
        IndexExpr::select(positionIE < 0, positionIE + boundIE, positionIE);
    auto outputVal = create.krnl.load(
        operandAdaptor.input_sequence(), positionIE.getValue());

    rewriter.replaceOp(op, outputVal);
    return success();
  }
};

void populateLoweringONNXSequenceAtOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceAtOpLowering>(typeConverter, ctx);
}
