/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Squeeze.cpp - Lowering Squeeze Op --------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Squeeze Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXSqueezeOpLowering : public ConversionPattern {
  ONNXSqueezeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSqueezeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeOpAdaptor operandAdaptor(operands);
    ONNXSqueezeOp squeezeOp = dyn_cast_or_null<ONNXSqueezeOp>(op);

    auto loc = op->getLoc();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto memRefShape = memRefType.getShape();
    auto elementSizeInBytes = getMemRefEltSizeInBytes(memRefType);
    Value data = operandAdaptor.data();

    ONNXSqueezeOpShapeHelper shapeHelper(&squeezeOp, rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));

    // Lower to ReinterpretCastOp so that the data is never copied or modified.
    Value newView = emitMemRefReinterpretCastOp(
        rewriter, loc, data, memRefType, shapeHelper.dimsForOutput(0));
    rewriter.replaceOp(op, newView);
    return success();
  }
};

void populateLoweringONNXSqueezeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSqueezeOpLowering>(ctx);
}
