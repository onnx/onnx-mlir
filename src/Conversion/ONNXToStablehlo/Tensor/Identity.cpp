/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Identity.cpp - Lowering Identity Op ----------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNXIdentity operator to the Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXIdentityOpLoweringToStablehlo : public ConversionPattern {
  ONNXIdentityOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXIdentityOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXIdentityOpAdaptor operandAdaptor(operands);
    rewriter.replaceOp(op, operandAdaptor.getInput());
    return success();
  }
};

} // namespace

void populateLoweringONNXIdentityOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXIdentityOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
