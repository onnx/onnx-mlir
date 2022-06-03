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

  StickViewUnstickRemovalPattern(MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/2) {}

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

class SetPrevLayerInLSTMOpPattern : public OpRewritePattern<ZLowLSTMOp> {
public:
  using OpRewritePattern<ZLowLSTMOp>::OpRewritePattern;

  // Set lower benefit than StickViewUnstickRemovalPattern's
  // to schedule this pattern after StickViewUnstickRemovalPattern.
  SetPrevLayerInLSTMOpPattern(MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(
      ZLowLSTMOp lstmOp, PatternRewriter &rewriter) const override {
    Value lstmInput = lstmOp.input();
    StringRef prevLayer = lstmOp.prev_layer();
    if (strcmp(prevLayer.data(), "not_set")) // if prev_layer is set already
      return failure();

    // Search for LSTM/GRU op that generates the input argument.
    StringRef directionAttr = "";
    for (Operation *user : lstmInput.getUsers()) {
      if (isa<ZLowLSTMOp>(user)) {
        ZLowLSTMOp userLstmOp = llvm::dyn_cast<ZLowLSTMOp>(user);
        if ((userLstmOp != lstmOp) &&
            ((userLstmOp.hn_output() == lstmInput) ||
                (userLstmOp.cf_output() == lstmInput))) {
          directionAttr = userLstmOp.direction();
          break;
        }
      }
      if (isa<ZLowGRUOp>(user)) {
        ZLowGRUOp userGruOp = llvm::dyn_cast<ZLowGRUOp>(user);
        if (userGruOp.hn_output() == lstmInput) {
          directionAttr = userGruOp.direction();
          break;
        }
      }
    }
    StringAttr prevLayerAttr;
    if (directionAttr.empty() || !strcmp(directionAttr.data(), "")) {
      prevLayerAttr = rewriter.getStringAttr("none");
    } else if (!strcmp(directionAttr.data(), "bidirectional")) {
      prevLayerAttr = rewriter.getStringAttr("bidir");
    } else {
      prevLayerAttr = rewriter.getStringAttr("uni");
    }
    // Update a zlow.LSTMOp operation.
    lstmOp.prev_layerAttr(prevLayerAttr);

    return success();
  }
};

class SetPrevLayerInGRUOpPattern : public OpRewritePattern<ZLowGRUOp> {
public:
  using OpRewritePattern<ZLowGRUOp>::OpRewritePattern;

  // Set lower benefit than StickViewUnstickRemovalPattern's
  // to schedule this pattern after StickViewUnstickRemovalPattern.
  SetPrevLayerInGRUOpPattern(MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(
      ZLowGRUOp gruOp, PatternRewriter &rewriter) const override {
    Value gruInput = gruOp.input();
    StringRef prevLayer = gruOp.prev_layer();
    if (strcmp(prevLayer.data(), "not_set")) // if prev_layer is set already
      return failure();

    // Search for LSTM/GRU op that generates the input argument.
    StringRef directionAttr = "";
    for (Operation *user : gruInput.getUsers()) {
      if (isa<ZLowGRUOp>(user)) {
        ZLowGRUOp userGruOp = llvm::dyn_cast<ZLowGRUOp>(user);
        if (userGruOp != gruOp) {
          directionAttr = userGruOp.direction();
          break;
        }
      }
      if (isa<ZLowLSTMOp>(user)) {
        directionAttr = llvm::dyn_cast<ZLowLSTMOp>(user).direction();
        break;
      }
    }
    StringAttr prevLayerAttr;
    if (directionAttr.empty() || !strcmp(directionAttr.data(), "")) {
      prevLayerAttr = rewriter.getStringAttr("none");
    } else if (!strcmp(directionAttr.data(), "bidirectional")) {
      prevLayerAttr = rewriter.getStringAttr("bidir");
    } else {
      prevLayerAttr = rewriter.getStringAttr("uni");
    }
    // Update a zlow.GRUOp operation.
    gruOp.prev_layerAttr(prevLayerAttr);

    return success();
  }
};

/*!
 *  Function pass that optimizes ZLowIR.
 */
class ZLowRewritePass
    : public PassWrapper<ZLowRewritePass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "zlow-rewrite"; }

  StringRef getDescription() const override { return "Rewrite ZLow Ops."; }

  void runOnOperation() override {
    Operation *function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<StickViewUnstickRemovalPattern>(&getContext());
    patterns.insert<SetPrevLayerInLSTMOpPattern>(&getContext());
    patterns.insert<SetPrevLayerInGRUOpPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> createZLowRewritePass() {
  return std::make_unique<ZLowRewritePass>();
}

} // namespace zlow
} // namespace onnx_mlir
