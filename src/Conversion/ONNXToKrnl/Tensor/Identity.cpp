//===----------------- Identity.cpp - Lowering Identity Op ----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Identity Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXIdentityOpLowering : public ConversionPattern {
  ONNXIdentityOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXIdentityOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Create init block if this is the first operation in the function.
    createInitState(rewriter, loc, op);

    ONNXIdentityOpAdaptor operandAdaptor(operands);
    rewriter.replaceOp(op, operandAdaptor.input());
    return success();
  }
};

void populateLoweringONNXIdentityOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXIdentityOpLowering>(ctx);
}
