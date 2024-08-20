/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Constant.cpp - Lowering Constant Op -----------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Constant Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXConstantOpLoweringToStablehlo : public ConversionPattern {
  ONNXConstantOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConstantOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXConstantOp constantOp = cast<ONNXConstantOp>(op);

    if (constantOp.getSparseValue().has_value())
      return constantOp.emitWarning("Only support dense values at this time");
    assert(constantOp.getValue().has_value() && "Value is not set");
    auto attr = constantOp.getValue().value();
    Value result = rewriter.create<stablehlo::ConstantOp>(
        loc, ElementsAttrBuilder::toDenseElementsAttr(
                 mlir::cast<ElementsAttr>(attr)));
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXConstantOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
