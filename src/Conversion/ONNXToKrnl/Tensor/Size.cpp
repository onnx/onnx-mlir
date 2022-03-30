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
  ONNXSizeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSizeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Location loc = op->getLoc();
    bool insertDealloc = checkInsertDealloc(op);
    ONNXSizeOp sizeOp = cast<ONNXSizeOp>(op);
    MultiDialectBuilder<KrnlBuilder> create(rewriter, loc);

    ONNXSizeOpAdaptor operandAdaptor(operands);
    Value data = operandAdaptor.data();
    ArrayRef<int64_t> dataShape = data.getType().cast<MemRefType>().getShape();
    Value resultOperand = sizeOp.size();
    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());

    Value alloc =
        (hasAllConstantDimensions(memRefType))
            ? insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc)
            : insertAllocAndDealloc(
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
      MemRefBuilder createMemRef(rewriter, loc);
      MathBuilder createMath(createMemRef);
      for (size_t i = 0; i < dataShape.size(); i++) {
        if (dataShape[i] == -1) {
          Value index = createMemRef.dim(data, i);
          Value dim = createMath.cast(memRefType.getElementType(), index);
          noElements = createMath.mul(noElements, dim);
        }
      }
    }

    create.krnl.store(noElements, alloc, llvm::None);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSizeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSizeOpLowering>(typeConverter, ctx);
}
