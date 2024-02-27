/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ShapeInferencePass.cpp - Shape Inference ---------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// ShapeInferencePass infers shapes for all operations and propagates shapes
// to their users and function signature return types.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/Transforms/ShapeInference.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ShapeInferencePass
    : public PassWrapper<ShapeInferencePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

  StringRef getArgument() const override { return "shape-inference"; }

  StringRef getDescription() const override { return "ONNX shape inference."; }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet cumulativePatterns(context);
    getShapeInferencePatterns(cumulativePatterns);
    patterns = FrozenRewritePatternSet(std::move(cumulativePatterns));
    return success();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(f.getBody(), patterns, config);
    inferFunctionReturnShapes(f);
  }

  FrozenRewritePatternSet patterns;
};

} // end anonymous namespace

/*!
 * Create a Shape Inference pass.
 */
std::unique_ptr<Pass> createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

} // namespace onnx_mlir
