/*
 * SPDX-License-Identifier: Apache-2.0
 */
//===-------SequenceInsert.cpp - Lowering SequenceInsert Op---------------=== //
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX SequenceInsert Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSequenceInsertOpLowering : public ConversionPattern {
  ONNXSequenceInsertOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXSequenceInsertOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXSequenceInsertOpAdaptor operandAdaptor(operands);
    ONNXSequenceInsertOp thisOp = dyn_cast<ONNXSequenceInsertOp>(op);
    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);
    IndexExprScope IEScope(&rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType =
        typeConverter->convertType(thisOp.getResult().getType());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();

    auto input_sequence = operandAdaptor.input_sequence();
    auto dimSize = create.mem.dim(input_sequence, 0);
    SymbolIndexExpr boundIE(dimSize);

    // Handle Optional and negative position
    IndexExpr positionIE;
    if (isFromNone(operandAdaptor.position())) {
      // Insert at the end of the sequence
      // Could be optimized as: Copy the input sequence and attach input tensor
      // at the end But the size for KrnlMemcpy is integer, not Value
      // memref::copy requires that source and destination have the same shape
      positionIE = boundIE;
    } else {
      positionIE = SymbolIndexExpr(create.krnl.load(operandAdaptor.position()));
      // Handle the negative position
      IndexExpr condIE = positionIE < 0;
      IndexExpr fixedPosition = positionIE + boundIE;
      positionIE = IndexExpr::select(condIE, fixedPosition, positionIE);
    }

    Value alloc = rewriter.create<KrnlSeqInsertOp>(loc, outputMemRefType,
        operandAdaptor.tensor(), operandAdaptor.input_sequence(),
        positionIE.getValue());

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSequenceInsertOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceInsertOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
