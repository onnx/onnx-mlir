//===---------------- Transpose.cpp - Lowering Transpose Op ---------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Transpose Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXTransposeOpLowering : public ConversionPattern {
  ONNXTransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTransposeOpAdaptor operandAdaptor(operands);
    ONNXTransposeOp transposeOp = llvm::cast<ONNXTransposeOp>(op);
    auto loc = op->getLoc();

    // Operands and attributes.
    Value data = operandAdaptor.data();
    auto permAttr = transposeOp.perm();

    // Basic information.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int64_t rank = memRefType.getShape().size();

    // Get a shape helper.
    ONNXTransposeOpShapeHelper shapeHelper(&transposeOp, &rewriter);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(0));

    // Create loop.
    BuildKrnlLoop inputLoops(rewriter, loc, rank);
    inputLoops.createDefineAndIterateOp(data);
    rewriter.setInsertionPointToStart(inputLoops.getIterateBlock());
    {
      // Get a child IndexExpr context.
      IndexExprContext childContext(shapeHelper.context);

      // Get read/write indices.
      SmallVector<IndexExpr, 4> readIndices;
      SmallVector<IndexExpr, 4> writeIndices;
      for (decltype(rank) i = 0; i < rank; ++i) {
        Value readVal = inputLoops.getInductionVar(i);
        Value writeVal =
            inputLoops.getInductionVar(ArrayAttrIntVal(permAttr, i));
        IndexExpr readIndex = childContext.createLoopInductionIndex(readVal);
        IndexExpr writeIndex = childContext.createLoopInductionIndex(writeVal);
        readIndices.emplace_back(readIndex);
        writeIndices.emplace_back(writeIndex);
      }

      // Copy data.
      Value loadData = childContext.createLoadOp(data, readIndices);
      childContext.createStoreOp(loadData, alloc, writeIndices);
    }

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXTransposeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLowering>(ctx);
}
