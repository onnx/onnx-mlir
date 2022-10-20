/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Identity.cpp - Lowering Identity Op ----------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNXIdentity operator to the Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXIdentityOpLoweringToMhlo : public ConversionPattern {
  ONNXIdentityOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXIdentityOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXIdentityOpAdaptor operandAdaptor(operands);
    rewriter.replaceOp(op, operandAdaptor.input());
    return success();
  }
};

} // namespace

void populateLoweringONNXIdentityOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXIdentityOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
