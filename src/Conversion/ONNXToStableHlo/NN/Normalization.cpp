/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Normalization.cpp - Lowering Normalization Ops -----------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers ONNX Normalization Operators to StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStableHlo/ONNXToStableHloCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXBatchNormalizationInferenceModeOpLoweringToStableHlo
    : public ConversionPattern {
  ONNXBatchNormalizationInferenceModeOpLoweringToStableHlo(MLIRContext *ctx)
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

    Value operand = operandAdaptor.getX();
    Value scale = operandAdaptor.getScale();
    Value bias = operandAdaptor.getB();
    Value mean = operandAdaptor.getMean();
    Value variance = operandAdaptor.getVar();
    llvm::APFloat eps = operandAdaptor.getEpsilon();

    Value result = rewriter.create<stablehlo::BatchNormInferenceOp>(loc,
        op->getResultTypes(), operand, scale, bias, mean, variance, eps, 1);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXNormalizationOpToStableHloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXBatchNormalizationInferenceModeOpLoweringToStableHlo>(
      ctx);
}

} // namespace onnx_mlir
