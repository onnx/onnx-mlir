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
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSequenceAtOpLowering : public OpConversionPattern<ONNXSequenceAtOp> {
  ONNXSequenceAtOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXSequenceAtOp seqOp,
      ONNXSequenceAtOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = seqOp.getOperation();
    Location loc = ONNXLoc<ONNXSequenceAtOp>(op);
    Value input_sequence = adaptor.getInputSequence();

    MultiDialectBuilder<KrnlBuilder, MemRefBuilder> create(rewriter, loc);
    IndexExprScope IEScope(&rewriter, loc);

    Type outputMemRefType =
        mlir::cast<MemRefType>(input_sequence.getType()).getElementType();

    auto dimSize = create.mem.dim(input_sequence, 0);
    SymbolIndexExpr boundIE(dimSize);
    IndexExpr positionIE = SymIE(create.krnl.load(adaptor.getPosition()));
    // Handle the negative position
    IndexExpr condIE = positionIE < 0;
    IndexExpr fixedPosition = positionIE + boundIE;
    positionIE = IndexExpr::select(condIE, fixedPosition, positionIE);
    Value positionVal = positionIE.getValue();

    Value outputVal = rewriter.create<KrnlSeqExtractOp>(loc, outputMemRefType,
        input_sequence, positionVal,
        IntegerAttr::get(rewriter.getIntegerType(1, false), 1));

    rewriter.replaceOp(op, outputVal);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXSequenceAtOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceAtOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
