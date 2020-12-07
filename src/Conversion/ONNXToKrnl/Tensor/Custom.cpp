//===----------------- Custom.cpp - Lowering Custom Op ----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Custom Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXCustomOpLowering : public ConversionPattern {
  ONNXCustomOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXCustomOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXCustomOpAdaptor operandAdaptor(operands);
    //rewriter.replaceOp(CallOp, operandAdaptor.input());
    auto funcAttr = op->getAttr("function_name");
    op->setAttr("callee",funcAttr);
    rewriter.replaceOpWithNewOp<CallOp>(
        op, op->getResult(0).getType(), operandAdaptor.input());
    return success();
  }
};

void populateLoweringONNXCustomOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXCustomOpLowering>(ctx);
}
