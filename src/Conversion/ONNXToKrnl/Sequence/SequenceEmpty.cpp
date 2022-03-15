/*
 * SPDX-License-Identifier: Apache-2.0
 */
//===-------SequenceEmpty.cpp - Lowering SequenceEmpty Op-----------------=== //
//
// Copyright 2022 The IBM Research Authors.
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

    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());

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
