/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Identity.cpp - Lowering Identity Op ----------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNXIdentity operator to the StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStableHlo/ONNXToStableHloCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXIdentityOpLoweringToStableHlo : public ConversionPattern {
  ONNXIdentityOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXIdentityOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXIdentityOpAdaptor operandAdaptor(operands);
    rewriter.replaceOp(op, operandAdaptor.getInput());
    return success();
  }
};

} // namespace

void populateLoweringONNXIdentityOpToStableHloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXIdentityOpLoweringToStableHlo>(ctx);
}

} // namespace onnx_mlir
