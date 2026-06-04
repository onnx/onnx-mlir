// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/// Remove redundant chained ONNXReluOps:
///   %1 = onnx.Relu(%0)
///   %2 = onnx.Relu(%1)
/// becomes:
///   %2 = onnx.Relu(%0)
///
/// Relu is idempotent (Relu(Relu(x)) == Relu(x)), so this bypasses one level
/// per application. The greedy driver re-runs until the whole chain collapses
/// and DCEs the now-dead inner Relus.
struct RemoveRedundantReluPattern : public OpRewritePattern<ONNXReluOp> {
  using OpRewritePattern<ONNXReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReluOp reluOp, PatternRewriter &rewriter) const override {
    auto prevRelu = reluOp.getX().getDefiningOp<ONNXReluOp>();
    if (!prevRelu)
      return failure();

    // Rebuild this Relu on the inner input and replace through the rewriter so
    // the ResultNamesUpdater listener transfers ResultNames onto the new op.
    auto newRelu = rewriter.create<ONNXReluOp>(
        reluOp.getLoc(), reluOp.getType(), prevRelu.getX());
    rewriter.replaceOp(reluOp, newRelu.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct RemoveRedundantReluPass
    : public PassWrapper<RemoveRedundantReluPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "remove-redundant-relu"; }
  StringRef getDescription() const override {
    return "Eliminate redundant chains of onnx.Relu operations";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RemoveRedundantReluPattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createRemoveRedundantReluPass() {
  return std::make_unique<RemoveRedundantReluPass>();
}

} // namespace onnx_mlir
