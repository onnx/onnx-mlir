/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXDecompose.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to decompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the decomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

// Create an DenseElementsAttr of ArrayAttr.
// This function is used to get Value Type of an EXISTING ArrayAttr for Scaler
// function.
DenseElementsAttr createDenseArrayAttr(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  if (origAttrs.getValue()[0].dyn_cast<FloatAttr>()) {
    mlir::Type elementType = rewriter.getF32Type();
    int nElements = origAttrs.getValue().size();
    SmallVector<float, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i) {
      wrapper[i] = origAttrs.getValue()[i].cast<FloatAttr>().getValueAsDouble();
    }
    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  } else if (origAttrs.getValue()[0].dyn_cast<IntegerAttr>()) {
    mlir::Type elementType = rewriter.getIntegerType(64);
    int nElements = origAttrs.getValue().size();
    SmallVector<int64_t, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i) {
      wrapper[i] = origAttrs.getValue()[i].cast<IntegerAttr>().getInt();
    }
    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  }
}

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Transform/ONNX/ONNXDecompose.inc"

namespace {

struct DecomposeONNXToONNXPass
    : public PassWrapper<DecomposeONNXToONNXPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void DecomposeONNXToONNXPass::runOnFunction() {
  auto function = getFunction();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXOpsDialect>();

  // These ops will be decomposed into other ONNX ops. Hence, they will not be
  // available after this pass.
  target.addIllegalOp<ONNXReduceL1Op>();
  target.addIllegalOp<ONNXReduceL2Op>();
  target.addIllegalOp<ONNXReduceLogSumOp>();
  target.addIllegalOp<ONNXReduceLogSumExpOp>();
  target.addIllegalOp<ONNXReduceSumSquareOp>();
  target.addIllegalOp<ONNXScalerOp>();
  target.addIllegalOp<ONNXLogSoftmaxOp>();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
} // end anonymous namespace

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> mlir::createDecomposeONNXToONNXPass() {
  return std::make_unique<DecomposeONNXToONNXPass>();
}
