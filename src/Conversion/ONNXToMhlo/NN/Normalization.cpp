/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Normalization.cpp - Lowering Normalization Ops -----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX Normalization Operators to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXBatchNormalizationInferenceModeOpLoweringToMhlo
    : public ConversionPattern {
  ONNXBatchNormalizationInferenceModeOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXBatchNormalizationInferenceModeOp::getOperationName(), 1,
            ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // batchnorm{epsilon}(x, scale, bias, mean, variance) =
    //      scale * (x - mean) / sqrt(variance + epsilon) + bias
    ONNXBatchNormalizationInferenceModeOpAdaptor operandAdaptor(
        operands, op->getAttrDictionary());
    Location loc = op->getLoc();

    Value operand = operandAdaptor.X();
    Value scale = operandAdaptor.scale();
    Value bias = operandAdaptor.B();
    Value mean = operandAdaptor.mean();
    Value variance = operandAdaptor.var();
    llvm::APFloat eps = operandAdaptor.epsilon();

    Value result = rewriter.create<mhlo::BatchNormInferenceOp>(loc,
        op->getResultTypes(), operand, scale, bias, mean, variance, eps, 1);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXNormalizationOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXBatchNormalizationInferenceModeOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
