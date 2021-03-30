/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Size.cpp - Lowering Size Op
//-------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Size Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXSizeOpLowering : public ConversionPattern {
  ONNXSizeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSizeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Location loc = op->getLoc();
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    ONNXSizeOp sizeOp = llvm::dyn_cast<ONNXSizeOp>(op);

    ONNXSizeOpAdaptor operandAdaptor(operands);
    Value data = operandAdaptor.data();
    ArrayRef<int64_t> dataShape = data.getType().cast<MemRefType>().getShape();
    Value resultOperand = sizeOp.size();
    ValueRange indices;
    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, {resultOperand});

    // Accumulate static dimensions first.
    int64_t staticNumElement = 1;
    bool allStaticDimensions = true;
    for (unsigned i = 0; i < dataShape.size(); i++) {
      if (dataShape[i] != -1)
        staticNumElement *= dataShape[i];
      else
        allStaticDimensions = false;
    }
    // Accumulate the remaining dimensions that are unknown.
    Value noElements = emitConstantOp(
        rewriter, loc, memRefType.getElementType(), staticNumElement);
    if (!allStaticDimensions) {
      for (unsigned i = 0; i < dataShape.size(); i++) {
        if (dataShape[i] == -1) {
          Value index = rewriter.create<DimOp>(loc, data, i);
          Value dim = rewriter.create<IndexCastOp>(
              loc, index, memRefType.getElementType());
          noElements = rewriter.create<MulIOp>(loc, noElements, dim);
        }
      }
    }

    rewriter.create<KrnlStoreOp>(loc, noElements, alloc, llvm::None);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSizeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSizeOpLowering>(ctx);
}
