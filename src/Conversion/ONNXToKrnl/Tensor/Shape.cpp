//===----------------Shape.cpp - Lowering Shape Op----------------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Shape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXShapeOpLowering : public ConversionPattern {
  ONNXShapeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXShapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXShapeOpAdaptor operandAdaptor(operands);
    ONNXShapeOp shapeOp = llvm::cast<ONNXShapeOp>(op);
    bool insertDealloc = checkInsertDealloc(op);
    Location loc = op->getLoc();
    // Get input data.
    Value data = operandAdaptor.data();
    ArrayRef<int64_t> dataShape = data.getType().cast<MemRefType>().getShape();
    unsigned dataRank = dataShape.size();

    Value resultOperand = shapeOp.shape();
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());

    Value alloc;
    if (hasAllConstantDimensions(outputMemRefType))
      alloc =
          insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          outputMemRefType, loc, rewriter, insertDealloc, {resultOperand});

    // Iterate along the data shape storing dim value to result.
    for (int i = 0; i < dataRank; i++) {
      // Create store index value.
      Value storeVal;
      Value storeIndex =
          emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);

      // Checking for dynamic dimensions.
      if (dataShape[i] == -1) {
        Value shapeVal = rewriter.create<DimOp>(loc, data, storeIndex);
        storeVal = rewriter.create<IndexCastOp>(
            loc, shapeVal, outputMemRefType.getElementType());
      } else
        storeVal = emitConstantOp(
            rewriter, loc, outputMemRefType.getElementType(), dataShape[i]);

      rewriter.create<StoreOp>(loc, storeVal, alloc, storeIndex);
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXShapeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXShapeOpLowering>(ctx);
}
