/*
 * SPDX-License-Identifier: Apache-2.0
 */
//===----------------SequenceAt.cpp - Lowering SequenceAt-----------------=== //
//
// Copyright 2022 The IBM Research Authors.
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

namespace onnx_mlir {

struct ONNXSequenceAtOpLowering : public ConversionPattern {
  ONNXSequenceAtOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXSequenceAtOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXSequenceAtOpAdaptor operandAdaptor(operands);
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder> create(rewriter, loc);
    IndexExprScope IEScope(&rewriter, loc);

    Value input_sequence = operandAdaptor.input_sequence();
    Type outputMemRefType =
        input_sequence.getType().cast<MemRefType>().getElementType();
    auto dimSize = create.mem.dim(input_sequence, 0);
    SymbolIndexExpr boundIE(dimSize);
    IndexExpr positionIE =
        SymbolIndexExpr(create.krnl.load(operandAdaptor.position()));
    // Handle the negative position
    IndexExpr condIE = positionIE < 0;
    IndexExpr fixedPosition = positionIE + boundIE;
    positionIE = IndexExpr::select(condIE, fixedPosition, positionIE);

    Value outputVal = rewriter.create<KrnlSeqExtractOp>(loc, outputMemRefType,
        input_sequence, positionIE.getValue(),
        IntegerAttr::get(rewriter.getIntegerType(1, false), 1));

    rewriter.replaceOp(op, outputVal);
    return success();
  }
};

void populateLoweringONNXSequenceAtOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceAtOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
