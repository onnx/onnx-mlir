/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------SequenceErase.cpp - Lowering SequenceErase Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
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

struct ONNXSequenceEraseOpLowering : public ConversionPattern {
  ONNXSequenceEraseOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSequenceEraseOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);

    ONNXSequenceEraseOpAdaptor operandAdaptor(operands);

    // Get element type for seq from the output
    ShapedType outputElementType = (*op->result_type_begin()).cast<SeqType>().getElementType();

    Type outputElementConvertedType;
    if (!outputElementType.hasRank()) {
      auto elementType = outputElementType.getElementType();
      outputElementConvertedType = UnrankedMemRefType::get(elementType, 0);
    } else {
      outputElementConvertedType = convertToMemRefType(outputElementType);
    }

    auto input_sequence = operandAdaptor.input_sequence();
    MemRefBoundsIndexCapture inputBounds(input_sequence);
      
    // Use memref with 0 element for empty sequence
    SmallVector<int64_t, 1> dims;
    dims.emplace_back(-1);
    llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());
 
    MemRefType outputMemRefType = MemRefType::get(shape, outputElementConvertedType);
    //bool insertDealloc = checkInsertDealloc(op);
    auto outputBound = inputBounds.getDim(0)-1;
    SmallVector<IndexExpr, 1> ubsIE;
    ubsIE.emplace_back(outputBound);
    Value alloc = insertAllocAndDeallocSimple(rewriter, op, outputMemRefType, loc, ubsIE);
  
    // Fill the output sequence

    IndexExpr positionIE;
    if (isFromNone(operandAdaptor.position())) {
      // Insert at the end of the sequence
      // Could be optimized as: Copy the input sequence and attach input tensor at the end
      // But the size for KrnlMemcpy is integer, not Value 
      // memref::copy requires that source and destination have the same shape
      positionIE = inputBounds.getDim(0) - 1;
    } else {
      positionIE = SymbolIndexExpr(operandAdaptor.position());
      // Handle the negative position
      positionIE = IndexExpr::select(positionIE < 0, positionIE + inputBounds.getDim(0), positionIE);
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
            auto element = createKrnl.load(operandAdaptor.input_sequence(), indicesLoopInd[0]);
            createKrnl.store(element, alloc, indicesLoopInd[0]);
         });

    // Copy after the insert
    SmallVector<IndexExpr, 1> lbs1;
    lbs1.emplace_back(positionIE+1);
    SmallVector<IndexExpr, 1> ubs1;
    ubs1.emplace_back(inputBounds.getDim(0));
    ValueRange secondLoopDef = createKrnl.defineLoops(1);
    create.krnl.iterateIE(secondLoopDef, secondLoopDef, lbs1, ubs1,
         [&](KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
            auto element = createKrnl.load(operandAdaptor.input_sequence(), indicesLoopInd[0]);
            auto outputIndex = create.math.sub(indicesLoopInd[0], create.math.constantIndex(1));
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
