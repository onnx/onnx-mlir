/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXShapeOpLowering : public ConversionPattern {
  ONNXShapeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXShapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get shape.
    ONNXShapeOpAdaptor operandAdaptor(operands);
    ONNXShapeOp shapeOp = llvm::dyn_cast<ONNXShapeOp>(op);
    Location loc = op->getLoc();
    ONNXShapeOpShapeHelper shapeHelper(&shapeOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed));

    // TODO: if the dimensions are known at compile time
    // (shapeHelper.dimsForOutput literal), then we could use a constant array.
    // Insert an allocation and deallocation for the output of this operation.
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = outputMemRefType.getElementType();
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Iterate along the data shape storing dim value to result.
    KrnlBuilder createKrnl(rewriter, loc);
    MathBuilder createMath(createKrnl);
    uint64_t dataRank = shapeHelper.selectedData.size();
    for (uint64_t i = 0; i < dataRank; ++i) {
      Value val = shapeHelper.selectedData[i].getValue();
      Value intVal = createMath.cast(elementType, val);
      createKrnl.storeIE(intVal, alloc, {LiteralIndexExpr(i)});
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXShapeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXShapeOpLowering>(ctx);
}
