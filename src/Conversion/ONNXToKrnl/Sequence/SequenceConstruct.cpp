/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------SequenceConstruct.cpp - Lowering SequenceConstruct Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX SequenceConstruct Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXSequenceConstructOpLowering : public ConversionPattern {
  ONNXSequenceConstructOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSequenceConstructOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXSequenceConstructOpAdaptor operandAdaptor(operands);
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
    IndexExprScope IEScope(&rewriter, loc);

    //ONNXSequenceConstructOp sequenceConstructOp = cast<ONNXSequenceConstructOp>(op);

    // Get element type for seq from the output
    ShapedType outputElementType = (*op->result_type_begin()).cast<SeqType>().getElementType();

    Type outputElementConvertedType;
    if (!outputElementType.hasRank()) {
      auto elementType = outputElementType.getElementType();
      outputElementConvertedType = UnrankedMemRefType::get(elementType, 0);
    } else {
      outputElementConvertedType = convertToMemRefType(outputElementType);
    }

    auto inputs = operandAdaptor.inputs();
      
    // Use memref with 0 element for empty sequence
    SmallVector<int64_t, 1> dims;
    dims.emplace_back(inputs.size());
    llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());
 
    MemRefType outputMemRefType = MemRefType::get(shape, outputElementConvertedType);
    bool insertDealloc = checkInsertDealloc(op);
    Value alloc = insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);
    
    // Fill the sequence
    for(uint64_t i = 0 ; i < inputs.size(); i++) {
      auto input = inputs[i];
      // ToDo: copy the memref?
      auto element = rewriter.create<memref::CastOp>(loc, outputElementConvertedType, input);
      create.krnl.store(element, alloc, create.math.constantIndex(i));
    }
    
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSequenceConstructOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceConstructOpLowering>(typeConverter, ctx);
}
