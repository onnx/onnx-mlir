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
    auto outputBound = boundIE + 1;

    Value alloc = rewriter.create<KrnlSeqAllocOp>(
        loc, outputMemRefType, outputBound.getValue());

    // Handle Optional and negative position
    IndexExpr positionIE;
    if (isFromNone(operandAdaptor.position())) {
      // Insert at the end of the sequence
      // Could be optimized as: Copy the input sequence and attach input tensor
      // at the end But the size for KrnlMemcpy is integer, not Value
      // memref::copy requires that source and destination have the same shape
      // ToDo (chentong): backward shape inference may help
      positionIE = boundIE;
    } else {
      positionIE = SymbolIndexExpr(create.krnl.load(operandAdaptor.position()));
      // Handle the negative position
      IndexExpr condIE = positionIE < 0;
      IndexExpr fixedPosition = positionIE + boundIE;
      positionIE = IndexExpr::select(condIE, fixedPosition, positionIE);
    }

    // Copy the elements before the position
    KrnlBuilder createKrnl(rewriter, loc);

    if (outputMemRefType.getShape()[0] == 1) {
      // This means the input sequence is empty.
      // No need to copy.
      // This test is essential because the empty sequence usually has
      // unranked tensor as element. The following loop body will have
      // compilation problem due to the unranked tensor even though
      // the loop will not be reached at runtime.
    } else {
      SmallVector<IndexExpr, 1> lbs;
      lbs.emplace_back(LiteralIndexExpr(0));
      SmallVector<IndexExpr, 1> ubs;
      ubs.emplace_back(positionIE);
      ValueRange firstLoopDef = createKrnl.defineLoops(1);
      createKrnl.iterateIE(firstLoopDef, firstLoopDef, lbs, ubs,
          [&](KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
            auto element = createKrnl.load(
                operandAdaptor.input_sequence(), indicesLoopInd[0]);
            createKrnl.seqstore(element, alloc, positionIE);
            // createKrnl.store(element, alloc, indicesLoopInd[0]);
          });

      // Copy the elements after the position
      SmallVector<IndexExpr, 1> lbs1;
      lbs1.emplace_back(positionIE + 1);
      SmallVector<IndexExpr, 1> ubs1;
      ubs1.emplace_back(boundIE);
      ValueRange secondLoopDef = createKrnl.defineLoops(1);
      createKrnl.iterateIE(secondLoopDef, secondLoopDef, lbs1, ubs1,
          [&](KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
            auto element = createKrnl.load(
                operandAdaptor.input_sequence(), indicesLoopInd[0]);
            auto oneIndex = create.math.constantIndex(1);
            auto outputIndex = create.math.add(indicesLoopInd[0], oneIndex);
            createKrnl.seqstore(element, alloc, outputIndex);
          });
    }

    // Insert the element at the position
    createKrnl.seqstore(operandAdaptor.tensor(), alloc, positionIE);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSequenceInsertOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceInsertOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
