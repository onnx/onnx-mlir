/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------SequenceEmpty.cpp - Lowering SequenceEmpty
//Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX SequenceEmpty Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXSequenceEmptyOpLowering : public ConversionPattern {
  ONNXSequenceEmptyOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXSequenceEmptyOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    // Get element type for seq from the output
    ShapedType outputElementType =
        (*op->result_type_begin()).cast<SeqType>().getElementType();
    auto elementType = outputElementType.getElementType();

    Type outputElementConvertedType;
    if (!outputElementType.hasRank()) {
      // convertToMemRefType can not handle unranked tensor
      outputElementConvertedType = UnrankedMemRefType::get(elementType, 0);
    } else {
      outputElementConvertedType = convertToMemRefType(outputElementType);
    }

    // Use memref with 0 element for empty sequence
    SmallVector<int64_t, 1> dims;
    dims.emplace_back(0);
    llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());

    MemRefType outputMemRefType =
        MemRefType::get(shape, outputElementConvertedType);
    bool insertDealloc = checkInsertDealloc(op);
    Value alloc =
        insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSequenceEmptyOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceEmptyOpLowering>(typeConverter, ctx);
}
