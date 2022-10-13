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
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSequenceEraseOpLowering : public ConversionPattern {
  ONNXSequenceEraseOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXSequenceEraseOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXSequenceEraseOpAdaptor operandAdaptor(operands);
    ONNXSequenceEraseOp thisOp = dyn_cast<ONNXSequenceEraseOp>(op);
    MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);
    IndexExprScope IEScope(&rewriter, loc);

    auto input_sequence = operandAdaptor.input_sequence();
    auto dimSize = create.mem.dim(input_sequence, 0);
    SymbolIndexExpr boundIE(dimSize);
    auto seqElementType =
        input_sequence.getType().cast<MemRefType>().getElementType();

    SmallVector<int64_t, 1> dims;
    // Number of element in seq may be statically known from shape inference
    dims.emplace_back(thisOp.getResult().getType().cast<SeqType>().getLength());
    llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());
    MemRefType outputMemRefType = MemRefType::get(shape, seqElementType);

    auto outputBound = boundIE - 1;
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
      positionIE = boundIE - 1;
    } else {
      positionIE = SymbolIndexExpr(operandAdaptor.position());
      // Handle the negative position
      auto correctionIE = positionIE + boundIE;
      positionIE = IndexExpr::select(positionIE < 0, correctionIE, positionIE);
    }

    // Copy the elements before the position
    KrnlBuilder createKrnl(rewriter, loc);
    SmallVector<IndexExpr, 1> lbs;
    lbs.emplace_back(LiteralIndexExpr(0));
    SmallVector<IndexExpr, 1> ubs;
    ubs.emplace_back(positionIE);
    ValueRange firstLoopDef = createKrnl.defineLoops(1);
    createKrnl.iterateIE(firstLoopDef, firstLoopDef, lbs, ubs,
        [&](KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
          auto element = createKrnl.load(
              operandAdaptor.input_sequence(), indicesLoopInd[0]);
          createKrnl.store(element, alloc, indicesLoopInd[0]);
        });

    // Free the element to be erased
    Value element =
        createKrnl.load(operandAdaptor.input_sequence(), positionIE.getValue());
    create.mem.dealloc(element);

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
          auto outputIndex = create.math.sub(indicesLoopInd[0], oneIndex);
          createKrnl.store(element, alloc, outputIndex);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSequenceEraseOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceEraseOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
