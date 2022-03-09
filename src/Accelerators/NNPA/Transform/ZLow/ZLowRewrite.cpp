/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZLowRewrite.cpp - ZLow Rewrite Patterns ---------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This pass implements optimizations for ZLow operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

/// This pattern rewrites
/// ```mlir
///   zlow.unstick(%input, %output)
///   %view = viewOp(%output)
///   zlow.stick(%view, %res)
/// ```
/// by removing `zlow.stick` and replacing `%res` by `%input`, which is
/// constrained by that `%input` and `%res` have the same static shape.
/// This pattern potentially removes `zlow.unstick` and `viewOp` if they are
/// dangling.
///
/// `viewOp` can be any op that inherits ViewLikeOpInterface, e.g.
/// memref.reinterpret_cast, memref.collapse_shape, memref.expand_shape.
//
class StickViewUnstickRemovalPattern : public OpRewritePattern<ZLowStickOp> {
public:
  using OpRewritePattern<ZLowStickOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZLowStickOp stickOp, PatternRewriter &rewriter) const override {
    Value stickInput = stickOp.X();

    // Input is a block argument, ignore it.
    if (stickInput.dyn_cast<BlockArgument>())
      return failure();

    // Input is a view.
    ViewLikeOpInterface viewOp =
        llvm::dyn_cast<ViewLikeOpInterface>(stickInput.getDefiningOp());
    if (!viewOp)
      return failure();
    // Get the source of the view.
    Value viewSource = viewOp.getViewSource();

    // Get UnstickOp that unstickifies the view source.
    // There is only one UnstickOp per buffer, so stop searching when we get
    // one.
    ZLowUnstickOp unstickOp;
    for (Operation *user : viewSource.getUsers()) {
      ZLowUnstickOp userOp = llvm::dyn_cast<ZLowUnstickOp>(user);
      if (!userOp)
        continue;
      // UnstickOp must be before the view operation.
      if (userOp.Out() == viewSource &&
          user->isBeforeInBlock(viewOp.getOperation())) {
        unstickOp = userOp;
        break;
      }
    }
    if (!unstickOp)
      return failure();

    // Match shapes.
    Value stickRes = stickOp.Out();
    Value unstickInput = unstickOp.X();
    MemRefType stickResType = stickRes.getType().dyn_cast<MemRefType>();
    MemRefType unstickInputType = unstickInput.getType().dyn_cast<MemRefType>();
    if (!stickResType.hasStaticShape() ||
        (stickResType.getShape() != unstickInputType.getShape()))
      return failure();

    // Rewrite
    rewriter.eraseOp(stickOp);
    stickRes.replaceAllUsesWith(unstickInput);
    // Remove the view op if there is no use.
    if (viewOp.getOperation()->getResults()[0].use_empty())
      rewriter.eraseOp(viewOp);
    // Remove unstick if there is no use of its second operand except itself.
    if (unstickOp.Out().hasOneUse())
      rewriter.eraseOp(unstickOp);

    return success();
  }
};

/*!
 *  Function pass that optimizes ZLowIR.
 */
class ZLowRewritePass
    : public PassWrapper<ZLowRewritePass, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const override { return "zlow-rewrite"; }

  StringRef getDescription() const override { return "Rewrite ZLow Ops."; }

  void runOnOperation() override {
    Operation *function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<StickViewUnstickRemovalPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> createZLowRewritePass() {
  return std::make_unique<ZLowRewritePass>();
}

} // namespace zlow
} // namespace onnx_mlir