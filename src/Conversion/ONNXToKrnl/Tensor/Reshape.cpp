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
    auto memRefType = convertToMemRefType(*op->result_type_begin());

    ONNXReshapeOpShapeHelper shapeHelper(&reshapeOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed));

    // Lower to ReinterpretCastOp so that the data is never copied or modified.
    Value newView = emitMemRefReinterpretCastOp(
        rewriter, loc, data, memRefType, shapeHelper.dimsForOutput(0));
    rewriter.replaceOp(op, newView);
    return success();
  }
};

void populateLoweringONNXReshapeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpLowering>(ctx);
}
