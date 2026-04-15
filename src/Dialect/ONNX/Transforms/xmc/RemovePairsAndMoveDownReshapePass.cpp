// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

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

    Operation *next = *reshape1->user_begin();
    if (!next)
      return failure();

    // Collect all XCOMPILERFusedEltwiseOps in the chain.
    SmallVector<Operation *> eltwiseChain;
    Operation *cursor = next;
    while (
        cursor && isa<XCOMPILERFusedEltwiseOp>(cursor) && cursor->hasOneUse()) {
      eltwiseChain.push_back(cursor);
      cursor = *cursor->user_begin();
    }
    if (eltwiseChain.empty())
      return failure();

    auto reshape2 = dyn_cast_or_null<ONNXReshapeOp>(cursor);
    if (!reshape2)
      return failure();

    auto reshape2OutTy =
        dyn_cast<RankedTensorType>(reshape2.getResult().getType());
    if (!reshape2OutTy || !reshape2OutTy.hasStaticShape())
      return failure();

    if (reshape1DataTy.getShape() != reshape2OutTy.getShape())
      return failure();

    // For binary eltwises, verify that input B is compatible with the
    // original (pre-reshape) shape. Two checks are needed:
    //
    // 1. Rank check: B must have the same rank as the original shape.
    //    Removing reshapes that change rank would produce a FusedEltwise
    //    whose broadcast output rank differs from reshape2's output rank,
    //    causing downstream shape-inference assertions.
    //
    // 2. Broadcast check: B must be broadcastable with the original shape.
    //    The reshape may merge spatial dims (e.g. [300,4] → [1,1200]),
    //    making B compatible with the reshaped A but incompatible with
    //    the original A (dim: 4 vs 1200).
    int64_t origRank = reshape1DataTy.getRank();
    ArrayRef<int64_t> origShape = reshape1DataTy.getShape();
    for (Operation *eltwiseOp : eltwiseChain) {
      auto fusedEltwise = cast<XCOMPILERFusedEltwiseOp>(eltwiseOp);
      Value inputB = fusedEltwise.getB();
      if (!isa<NoneType>(inputB.getType())) {
        auto inputBTy = dyn_cast<RankedTensorType>(inputB.getType());
        if (inputBTy && inputBTy.getRank() != origRank)
          return failure();
        if (inputBTy) {
          SmallVector<int64_t> broadcastResult;
          if (!OpTrait::util::getBroadcastedShape(
                  origShape, inputBTy.getShape(), broadcastResult))
            return failure();
        }
      }
    }

    // Remove reshape1, update eltwise result types in the chain, remove
    // reshape2. Preserve each eltwise's original element type (which may
    // be quantized) — only update the shape.
    rewriter.replaceOp(reshape1, reshape1Data);
    for (Operation *eltwiseOp : eltwiseChain) {
      rewriter.modifyOpInPlace(eltwiseOp, [&]() {
        auto curType =
            cast<RankedTensorType>(eltwiseOp->getResult(0).getType());
        auto newType = RankedTensorType::get(
            reshape2OutTy.getShape(), curType.getElementType());
        eltwiseOp->getResult(0).setType(newType);
      });
    }
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
    ResultNamesUpdater rnUpdater;
    config.maxIterations = 10;
    config.useTopDownTraversal = true;
    config.listener = &rnUpdater;

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createRemovePairsAndMoveDownReshapePass() {
  return std::make_unique<RemovePairsAndMoveDownReshapePass>();
}

} // namespace onnx_mlir
