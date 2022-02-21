/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------SequenceEmpty.cpp - Lowering SequenceEmpty Op----------------------=== //
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
      : ConversionPattern(
            typeConverter, mlir::ONNXSequenceEmptyOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    SmallVector<int64_t, 1> dims;
    dims.emplace_back(0);
    llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());
 
    //auto outputTensorType = RankedTensorType::get(shape, rewriter.getIntegerType(64));
    //auto outputMemRefType = convertToMemRefType(outputTensorType);
    MemRefType outputMemRefType = MemRefType::get(shape, rewriter.getIntegerType(64));
    bool insertDealloc = checkInsertDealloc(op);
    Value alloc = insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSequenceEmptyOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceEmptyOpLowering>(typeConverter, ctx);
}
