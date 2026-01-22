// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/// Remove redundant chained ONNXReluOps:
///   %1 = onnx.Relu(%0)
///   %2 = onnx.Relu(%1)
/// becomes:
///   %2 = onnx.Relu(%0)
struct RemoveRedundantReluPattern : public OpRewritePattern<ONNXReluOp> {
  using OpRewritePattern<ONNXReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXReluOp reluOp,
      PatternRewriter &rewriter) const override {
    Value inputVal = reluOp.getX();
    auto prevRelu = inputVal.getDefiningOp<ONNXReluOp>();
    if (!prevRelu)
      return failure();

    // Find the head of the chain (closest to original non-Relu input).
    auto headRelu = prevRelu;
    while (auto maybePrev = headRelu.getX().getDefiningOp<ONNXReluOp>()) {
      if (!headRelu->hasOneUse())
        break;
      headRelu = maybePrev;
    }

    // Find the tail (last Relu in chain without a Relu user).
    auto tailRelu = reluOp;
    bool extended = true;
    while (extended) {
      extended = false;
      if (!tailRelu->hasOneUse())
        break;
      for (auto *user : tailRelu.getResult().getUsers()) {
        if (auto next = dyn_cast<ONNXReluOp>(user)) {
          tailRelu = next;
          extended = true;
          break;
        }
      }
    }

    // Create a single Relu that directly consumes the head input.
    rewriter.setInsertionPointAfter(tailRelu);
    auto newRelu = rewriter.create<ONNXReluOp>(
        tailRelu.getLoc(), tailRelu.getType(), headRelu.getX());

    tailRelu.getResult().replaceAllUsesWith(newRelu.getResult());

    // Erase the old chain from tail backwards.
    auto cur = tailRelu;
    while (true) {
      auto prev = cur.getX().getDefiningOp<ONNXReluOp>();
      rewriter.eraseOp(cur);
      if (!prev)
        break;
      cur = prev;
    }

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
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createRemoveRedundantReluPass() {
  return std::make_unique<RemoveRedundantReluPass>();
}

} // namespace onnx_mlir

