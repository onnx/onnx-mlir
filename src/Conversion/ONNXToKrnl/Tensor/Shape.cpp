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
      // Unknown dimension for ShapeOp can only come from unshaped tensor
      // 'tensor<*xT>' type, which is not handled in the system yet.
      llvm_unreachable("unknown dimension for ShapeOp is not supported yet");

    // Iterate along the data shape storing dim value to result.
    for (int i = 0; i < dataRank; i++) {
      IndexExprContext IEContext(&rewriter, loc);
      IndexExpr storeIndex = IEContext.createLiteralIndex(i);
      IndexExpr shapeVal = IEContext.createDimIndexFromShapedType(data, i);
      Value storeVal = rewriter.create<IndexCastOp>(
            loc, shapeVal.getValue(), outputMemRefType.getElementType());
      rewriter.create<StoreOp>(loc, storeVal, alloc, storeIndex.getValue());
#if 0
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
#endif
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXShapeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXShapeOpLowering>(ctx);
}
