/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ZHighToONNX.cpp - ONNX dialect to ZHigh lowering -------------===//
//
// Copyright 2023- The IBM Research Authors.
//
// =============================================================================
//
// This file defines patterns to reconstruct ONNX ops from ZHigh ops.
//
// After all optimizations, if there are still light-weight ops (e.g. add,
// sub, ...) that are of `stick -> light-weight op -> unstick`, it's better to
// use CPU instead of NNPA to avoid stick/unstick. CPU is efficient to handle
// these ops, e.g. by vectorizing the computation.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ZHigh to ONNX Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXZHighToONNX.inc"

struct ZHighToONNXLoweringPass
    : public PassWrapper<ZHighToONNXLoweringPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHighToONNXLoweringPass)

  StringRef getArgument() const override { return "convert-zhigh-to-onnx"; }

  StringRef getDescription() const override {
    return "Reconstruct ONNX ops from ZHigh ops.";
  }

  void runOnOperation() final;
};
} // end anonymous namespace.

void ZHighToONNXLoweringPass::runOnOperation() {
  Operation *function = getOperation();
  ConversionTarget target(getContext());

  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  zhigh::ZHighStickOp::getCanonicalizationPatterns(patterns, &getContext());
  zhigh::ZHighUnstickOp::getCanonicalizationPatterns(patterns, &getContext());

  (void)applyPatternsAndFoldGreedily(function, std::move(patterns));
}

std::unique_ptr<Pass> createZHighToONNXPass() {
  return std::make_unique<ZHighToONNXLoweringPass>();
}

} // namespace onnx_mlir
