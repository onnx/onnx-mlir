/*
 * SPDX-License-Identifier: Apache-2.0
 */
//===-------SequenceErase.cpp - Lowering SequenceErase Op-----------------=== //
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX SequenceErase Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSequenceEraseOpLowering
    : public OpConversionPattern<ONNXSequenceEraseOp> {
  ONNXSequenceEraseOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXSequenceEraseOp seqOp,
      ONNXSequenceEraseOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = seqOp.getOperation();
    Location loc = ONNXLoc<ONNXSequenceEraseOp>(op);

    // This Op creates a new sequence from the input sequence
    // with the element at the specified position erased.
    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);
    IndexExprScope IEScope(&rewriter, loc);

    Value input_sequence = adaptor.getInputSequence();
    Value dimSize = create.mem.dim(input_sequence, 0);
    SymbolIndexExpr boundIE(dimSize);

    MemRefType outputMemRefType = mlir::cast<MemRefType>(
        typeConverter->convertType(seqOp.getResult().getType()));

    SymbolIndexExpr outputBound = boundIE - 1;
    Value outputBoundVal = outputBound.getValue();
    Value alloc =
        rewriter.create<KrnlSeqAllocOp>(loc, outputMemRefType, outputBoundVal);

    // Fill the output sequence

    IndexExpr positionIE;
    if (isNoneValue(adaptor.getPosition())) {
      // Erase the end of the sequence
      positionIE = boundIE - 1;
    } else {
      positionIE = SymIE(create.krnl.load(adaptor.getPosition()));
      // Handle the negative position
      IndexExpr correctionIE = positionIE + boundIE;
      IndexExpr conditionIE = positionIE < 0;
      positionIE = IndexExpr::select(conditionIE, correctionIE, positionIE);
    }

    // Copy the elements before the position
    KrnlBuilder createKrnl(rewriter, loc);
    createKrnl.forLoopIE(LitIE(0), positionIE, /*step*/ 1, /*par*/ false,
        [&](const KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
          Value element =
              createKrnl.load(adaptor.getInputSequence(), indicesLoopInd[0]);
          createKrnl.seqstore(element, alloc, positionIE);
          // createKrnl.store(element, alloc, indicesLoopInd[0]);
        });

    // Copy the elements after the position
    createKrnl.forLoopIE(positionIE + 1, boundIE, /*step*/ 1, /*par*/ false,
        [&](const KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
          Value element =
              createKrnl.load(adaptor.getInputSequence(), indicesLoopInd[0]);
          Value oneIndex = create.math.constantIndex(1);
          Value outputIndex = create.math.sub(indicesLoopInd[0], oneIndex);
          createKrnl.seqstore(element, alloc, outputIndex);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXSequenceEraseOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceEraseOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
