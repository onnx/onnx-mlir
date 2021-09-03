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

struct ONNXSqueezeV11OpLowering : public ConversionPattern {
  ONNXSqueezeV11OpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSqueezeV11Op::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSqueezeV11OpAdaptor operandAdaptor(operands);
    ONNXSqueezeV11Op squeezeOp = dyn_cast_or_null<ONNXSqueezeV11Op>(op);

    auto loc = op->getLoc();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Value data = operandAdaptor.data();

    ONNXSqueezeV11OpShapeHelper shapeHelper(&squeezeOp, rewriter,
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

void populateLoweringONNXSqueezeV11OpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSqueezeV11OpLowering>(ctx);
}
