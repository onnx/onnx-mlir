/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Hardmax.cpp - Hardmax Op ---------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

struct ONNXHardmaxOpLowering : public ConversionPattern {
  ONNXHardmaxOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXHardmaxOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXHardmaxOpAdaptor operandAdaptor(operands);

    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());
    auto elementType = memRefType.getElementType();

    int64_t rank = memRefType.getRank();
    int64_t axis = llvm::dyn_cast<ONNXHardmaxOp>(op).axis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);

    Value input = operandAdaptor.input();
    MemRefBoundsIndexCapture inputBounds(input);
    SmallVector<IndexExpr, 4> ubs;
    inputBounds.getDimList(ubs);

    // Insert an allocation and deallocation for the result of this operation.
    bool insertDealloc = checkInsertDealloc(op);
    Value resMemRef = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, ubs, insertDealloc);

    rewriter.replaceOp(op, resMemRef);
    return success();
  }
};

void populateLoweringONNXHardmaxOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXHardmaxOpLowering>(ctx);
}
