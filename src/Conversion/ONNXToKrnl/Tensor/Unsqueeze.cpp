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

struct ONNXUnsqueezeOpLowering : public ConversionPattern {
  ONNXUnsqueezeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXUnsqueezeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXUnsqueezeOpAdaptor operandAdaptor(operands);
    ONNXUnsqueezeOp unsqueezeOp = dyn_cast_or_null<ONNXUnsqueezeOp>(op);

    auto loc = op->getLoc();
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int outRank = memRefType.getRank();
    Value data = operandAdaptor.data();

    ONNXUnsqueezeOpShapeHelper shapeHelper(&unsqueezeOp, rewriter,
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

void populateLoweringONNXUnsqueezeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXUnsqueezeOpLowering>(ctx);
}
