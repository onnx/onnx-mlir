/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Constant.cpp - Lowering Constant Op -----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Constant Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXConstantOpLoweringToMhlo : public ConversionPattern {
  ONNXConstantOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConstantOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(StringAttr::get(op->getContext(),
                                    mlir::ONNXConstantOp::getOperationName()),
        op->getLoc());
    ONNXConstantOp constantOp = cast<ONNXConstantOp>(op);

    if (constantOp.sparse_value().hasValue())
      return constantOp.emitWarning("Only support dense values at this time");
    assert(constantOp.value().hasValue() && "Value is not set");
    Value result =
        rewriter.create<mhlo::ConstantOp>(loc, constantOp.value().getValue());
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXConstantOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
