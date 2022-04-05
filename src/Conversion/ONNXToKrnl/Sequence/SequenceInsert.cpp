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

    auto outputMemRefType = convertToMemRefType(thisOp.getResult().getType());
    auto seqElementConvertedType =
        outputMemRefType.getElementType().cast<MemRefType>();
    auto input_sequence = operandAdaptor.input_sequence();
    auto dimSize = create.mem.dim(input_sequence, 0);
    SymbolIndexExpr boundIE(dimSize);

    // Output sequence has one more element
    auto outputBound = boundIE + 1;
    SmallVector<IndexExpr, 1> ubsIE;
    ubsIE.emplace_back(outputBound);
    Value alloc =
        insertAllocAndDeallocSimple(rewriter, op, outputMemRefType, loc, ubsIE);

    // Fill the output sequence

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
      positionIE =
          IndexExpr::select(positionIE < 0, positionIE + boundIE, positionIE);
    }

    // Copy elements before the insertion position
    SmallVector<IndexExpr, 1> lbs;
    lbs.emplace_back(LiteralIndexExpr(0));
    SmallVector<IndexExpr, 1> ubs;
    ubs.emplace_back(positionIE);
    ValueRange firstLoopDef = create.krnl.defineLoops(1);
    create.krnl.iterateIE(firstLoopDef, firstLoopDef, lbs, ubs,
        [&](KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
          auto element = createKrnl.load(
              operandAdaptor.input_sequence(), indicesLoopInd[0]);
          auto converted = create.mem.cast(element, seqElementConvertedType);
          createKrnl.store(converted, alloc, indicesLoopInd[0]);
        });

    // Insert the input tensor
    // ToDo (chentong): need to duplicate the tensor
    auto element =
        create.mem.cast(operandAdaptor.tensor(), seqElementConvertedType);
    create.krnl.store(element, alloc, positionIE.getValue());

    // Copy elements after the insertion position
    SmallVector<IndexExpr, 1> lbs1;
    lbs1.emplace_back(positionIE + 1);
    SmallVector<IndexExpr, 1> ubs1;
    ubs1.emplace_back(outputBound);
    ValueRange secondLoopDef = create.krnl.defineLoops(1);
    create.krnl.iterateIE(secondLoopDef, secondLoopDef, lbs1, ubs1,
        [&](KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
          auto element = createKrnl.load(
              operandAdaptor.input_sequence(), indicesLoopInd[0]);
          auto converted = create.mem.cast(element, seqElementConvertedType);
          auto outputIndex =
              create.math.add(indicesLoopInd[0], create.math.constantIndex(1));
          createKrnl.store(converted, alloc, outputIndex);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSequenceInsertOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceInsertOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
