// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

static bool isReshapePairIntermediateOp(Operation *op) {
  if (!op)
    return false;
  // Match only the XCompiler custom eltwise op.
  return isa<XCOMPILERFusedEltwiseOp>(op);
}

/// Remove paired reshapes:
///   reshape1 -> (allowed ops)* -> reshape2
/// when:
///   shape(reshape1.data) == shape(reshape2.result)
struct RemovePairedReshapePattern : public OpRewritePattern<ONNXReshapeOp> {
  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReshapeOp reshape1, PatternRewriter &rewriter) const override {
    if (!reshape1->hasOneUse())
      return failure();

    Value reshape1Data = reshape1.getData();
    auto reshape1DataTy = dyn_cast<RankedTensorType>(reshape1Data.getType());
    if (!reshape1DataTy || !reshape1DataTy.hasStaticShape())
      return failure();

    Operation *next = *reshape1.getResult().getUsers().begin();
    if (!next)
      return failure();

    auto hasSingleLinearUse = [](Operation *op) -> bool {
      return op && op->getNumResults() == 1 && op->getResult(0).hasOneUse();
    };
    if (!hasSingleLinearUse(next))
      return failure();

    Operation *cursor = next;
    while (cursor && !isa<ONNXReshapeOp>(cursor) &&
           isReshapePairIntermediateOp(cursor)) {
      if (!hasSingleLinearUse(cursor))
        return failure();
      cursor = *cursor->getResult(0).getUsers().begin();
    }

    auto reshape2 = dyn_cast_or_null<ONNXReshapeOp>(cursor);
    if (!reshape2)
      return failure();

    auto reshape2OutTy =
        dyn_cast<RankedTensorType>(reshape2.getResult().getType());
    if (!reshape2OutTy || !reshape2OutTy.hasStaticShape())
      return failure();

    if (reshape1DataTy.getShape() != reshape2OutTy.getShape())
      return failure();

    rewriter.replaceOp(reshape1, reshape1Data);
    rewriter.modifyOpInPlace(next, [=]() {
      next->getResult(0).setType(reshape2->getResult(0).getType());
    });
    rewriter.replaceOp(reshape2, reshape2.getData());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct RemovePairsAndMoveDownReshapePass
    : public PassWrapper<RemovePairsAndMoveDownReshapePass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "remove-pairs-and-move-down-reshape";
  }
  StringRef getDescription() const override {
    return "Remove paired reshapes across small XCompiler qlinear chains";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RemovePairedReshapePattern>(context);

    GreedyRewriteConfig config;
    config.maxIterations = 10;
    config.useTopDownTraversal = true;

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createRemovePairsAndMoveDownReshapePass() {
  return std::make_unique<RemovePairsAndMoveDownReshapePass>();
}

} // namespace onnx_mlir
