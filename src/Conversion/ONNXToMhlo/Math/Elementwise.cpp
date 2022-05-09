/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

template <>
struct MhloDialectOp<ONNXAddOp> {
  using Op = mhlo::AddOp;
};

// Element-wise variadic ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLoweringToMhlo : public ConversionPattern {
  ONNXElementwiseVariadicOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(ElementwiseVariadicOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(StringAttr::get(op->getContext(),
                                    ElementwiseVariadicOp::getOperationName()),
        op->getLoc());

    // TODO: check whether ONNX dialect has explicit broadcast feature
    auto mhloOp = rewriter.create<MhloOp<ElementwiseVariadicOp>>(
        loc, op->getResultTypes(), op->getOperands());
    rewriter.replaceOp(op, mhloOp.result());

    return success();
  }
};

void populateLoweringONNXElementwiseOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseVariadicOpLoweringToMhlo<ONNXAddOp>>(ctx);
}

} // namespace onnx_mlir
