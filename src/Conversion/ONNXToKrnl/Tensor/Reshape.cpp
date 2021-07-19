/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Reshape.cpp - Lowering Reshape Op -------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reshape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXReshapeOpLowering : public ConversionPattern {
  ONNXReshapeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXReshapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXReshapeOpAdaptor operandAdaptor(operands);
    ONNXReshapeOp reshapeOp = dyn_cast_or_null<ONNXReshapeOp>(op);

    auto loc = op->getLoc();
    Value data = operandAdaptor.data();
    Value shape = operandAdaptor.shape();
    auto dataShape = data.getType().cast<MemRefType>().getShape();
    auto memRefType = convertToMemRefType(*op->result_type_begin());

    ONNXReshapeOpShapeHelper shapeHelper(&reshapeOp, rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));

    // If the output shape is a constant, lower to ReinterpretCastOp so that the
    // data is never copied or modified.
    if (memRefType.hasStaticShape()) {
      Value newView = emitMemRefReinterpretCastOp(
          rewriter, loc, data, memRefType, shapeHelper.dimsForOutput(0));
      rewriter.replaceOp(op, newView);
      return success();
    }

    // Other cases, we have to do data copy.
    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(0));

    // Compute size in bytes using the input tensor.
    IndexExpr sizeInBytes =
        LiteralIndexExpr(getMemRefEltSizeInBytes(memRefType));
    sizeInBytes = sizeInBytes * shapeHelper.numOfElements;
    Value sizeInBytesI64 = rewriter.create<IndexCastOp>(
        loc, sizeInBytes.getValue(), rewriter.getI64Type());

    // Emit memcpy op.
    rewriter.create<KrnlMemcpyOp>(loc, alloc, data, sizeInBytesI64);
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXReshapeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpLowering>(ctx);
}
