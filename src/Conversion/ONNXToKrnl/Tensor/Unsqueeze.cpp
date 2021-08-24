/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Unsqueeze.cpp - Lowering Unsqueeze Op ----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Unsqueeze Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXUnsqueezeV11OpLowering : public ConversionPattern {
  ONNXUnsqueezeV11OpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXUnsqueezeV11Op::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXUnsqueezeV11OpAdaptor operandAdaptor(operands);
    ONNXUnsqueezeV11Op unsqueezeOp = dyn_cast_or_null<ONNXUnsqueezeV11Op>(op);

    auto loc = op->getLoc();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int outRank = memRefType.getRank();
    Value data = operandAdaptor.data();

    ONNXUnsqueezeV11OpShapeHelper shapeHelper(&unsqueezeOp, rewriter,
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

void populateLoweringONNXUnsqueezeV11OpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXUnsqueezeV11OpLowering>(ctx);
}
