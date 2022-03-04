/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------SequenceInsert.cpp - Lowering SequenceInsert Op---------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
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

struct ONNXSequenceInsertOpLowering : public ConversionPattern {
  ONNXSequenceInsertOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXSequenceInsertOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXSequenceInsertOpAdaptor operandAdaptor(operands);
    ONNXSequenceInsertOp thisOp = dyn_cast<ONNXSequenceInsertOp>(op);
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
    IndexExprScope IEScope(&rewriter, loc);

    // ONNXSequenceInsertOp sequenceInsertOp = cast<ONNXSequenceInsertOp>(op);

    // Get element type for seq from the output

    auto input_sequence = operandAdaptor.input_sequence();
    auto dimSize = rewriter.create<memref::DimOp>(
        loc, input_sequence, create.math.constantIndex(0));
    SymbolIndexExpr boundIE(dimSize);

    // Can not use this type because the output element type may be more precise
    // This issue may be resolved in shape inference by backwards propagation
    // In some sense, the type info for seq is not consistent at SequenceInsert
    // The code of type converter for Seq is duplicated
    // auto seqElementType =
    // input_sequence.getType().cast<MemRefType>().getElementType();

    ShapedType seqElementType =
        thisOp.getResult().getType().cast<SeqType>().getElementType();
    Type elementType = seqElementType.getElementType();
    Type seqElementConvertedType;
    if (seqElementType.hasRank()) {
      seqElementConvertedType =
          MemRefType::get(seqElementType.getShape(), elementType);
    } else {
      seqElementConvertedType = UnrankedMemRefType::get(elementType, 0);
    }

    SmallVector<int64_t, 1> dims;
    // Number of element in seq my be statically known from shape inference
    dims.emplace_back(thisOp.getResult().getType().cast<SeqType>().getLength());
    llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());
    MemRefType outputMemRefType =
        MemRefType::get(shape, seqElementConvertedType);

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

    // Copy before the insert
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
          auto converted = rewriter.create<memref::CastOp>(
              loc, seqElementConvertedType, element);
          createKrnl.store(converted, alloc, indicesLoopInd[0]);
        });

    // Insert the input tensor
    // ToDo: need to copy the tensor?
    auto element = rewriter.create<memref::CastOp>(
        loc, seqElementConvertedType, operandAdaptor.tensor());
    create.krnl.store(element, alloc, positionIE.getValue());

    // Copy after the insert
    SmallVector<IndexExpr, 1> lbs1;
    lbs1.emplace_back(positionIE + 1);
    SmallVector<IndexExpr, 1> ubs1;
    ubs1.emplace_back(outputBound);
    ValueRange secondLoopDef = createKrnl.defineLoops(1);
    create.krnl.iterateIE(secondLoopDef, secondLoopDef, lbs1, ubs1,
        [&](KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
          auto element = createKrnl.load(
              operandAdaptor.input_sequence(), indicesLoopInd[0]);
          auto converted = rewriter.create<memref::CastOp>(
              loc, seqElementConvertedType, element);
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
