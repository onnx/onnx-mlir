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

static void setPrevLayerAttrOfGeneratorZLowStickForLSTMOp(
    Value val, StringAttr prevLayerAttr) {
  for (Operation *user : val.getUsers()) {
    if (isa<ZLowStickForLSTMOp>(user)) {
      ZLowStickForLSTMOp stickOp = llvm::dyn_cast<ZLowStickForLSTMOp>(user);
      if (stickOp.out() == val) {
        stickOp.prev_layerAttr(prevLayerAttr);
        return;
      }
    }
  }
  return;
}

static void setPrevLayerAttrOfGeneratorZLowStickForGRUOp(
    Value val, StringAttr prevLayerAttr) {
  for (Operation *user : val.getUsers()) {
    if (isa<ZLowStickForGRUOp>(user)) {
      ZLowStickForGRUOp stickOp = llvm::dyn_cast<ZLowStickForGRUOp>(user);
      if (stickOp.out() == val) {
        stickOp.prev_layerAttr(prevLayerAttr);
      }
    }
  }
  return;
}

static void updatePrevLayerAttrs(
    Operation *op, Value input, PatternRewriter &rewriter) {
  // Check if op's input equals input.
  Value lstmGruInput;
  if (isa<ZLowLSTMOp>(op))
    lstmGruInput = llvm::dyn_cast<ZLowLSTMOp>(op).input();
  else if (isa<ZLowGRUOp>(op))
    lstmGruInput = llvm::dyn_cast<ZLowGRUOp>(op).input();
  else
    return;
  if (lstmGruInput != input) {
    return;
  }

  // Search for zlow.lstm/gru op that generates the input argument.
  StringRef directionAttr = "";
  for (Operation *user : lstmGruInput.getUsers()) {
    if (isa<ZLowLSTMOp>(user)) {
      ZLowLSTMOp userLstmOp = llvm::dyn_cast<ZLowLSTMOp>(user);
      if ((userLstmOp.hn_output() == lstmGruInput) ||
          (userLstmOp.cf_output() == lstmGruInput)) {
        directionAttr = userLstmOp.direction();
        break;
      }
    }
    if (isa<ZLowGRUOp>(user)) {
      ZLowGRUOp userGruOp = llvm::dyn_cast<ZLowGRUOp>(user);
      if (userGruOp.hn_output() == lstmGruInput) {
        directionAttr = userGruOp.direction();
        break;
      }
    }
  }
  StringAttr prevLayerAttr;
  if (directionAttr.empty() || !strcmp(directionAttr.data(), ""))
    prevLayerAttr = rewriter.getStringAttr("none");
  else if (!strcmp(directionAttr.data(), "bidirectional"))
    prevLayerAttr = rewriter.getStringAttr("bidir");
  else
    prevLayerAttr = rewriter.getStringAttr("uni");

  // Update prev_flag attribute of zlow.lstm/gru operation, and
  // zlow.StickForLSTM/GRU operations generating input of the zlow.lstm/gru.
  if (isa<ZLowLSTMOp>(op)) {
    ZLowLSTMOp lstmOp = llvm::dyn_cast<ZLowLSTMOp>(op);
    lstmOp.prev_layerAttr(prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForLSTMOp(
        lstmOp.input_weights(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForLSTMOp(
        lstmOp.input_bias(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForLSTMOp(
        lstmOp.hidden_weights(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForLSTMOp(
        lstmOp.hidden_bias(), prevLayerAttr);
  } else if (isa<ZLowGRUOp>(op)) {
    ZLowGRUOp gruOp = llvm::dyn_cast<ZLowGRUOp>(op);
    gruOp.prev_layerAttr(prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForGRUOp(
        gruOp.input_weights(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForGRUOp(
        gruOp.input_bias(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForGRUOp(
        gruOp.hidden_weights(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForGRUOp(
        gruOp.hidden_bias(), prevLayerAttr);
  }
}

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

    // Update the prev_layer attribute in zlow.lstm/gru operations using
    // "stickRes" as their inputs, and update the prev_layer attribute
    // in zlow.stickForLSTM/GRU generating inputs of the zlow.lstm/gru.
    for (Operation *user : unstickInput.getUsers()) {
      if (isa<ZLowLSTMOp>(user) || isa<ZLowGRUOp>(user)) {
        updatePrevLayerAttrs(user, unstickInput, rewriter);
      }
    }

    return success();
  }
};

class SetPrevLayerInLSTMOpPattern : public OpRewritePattern<ZLowLSTMOp> {
public:
  using OpRewritePattern<ZLowLSTMOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZLowLSTMOp lstmOp, PatternRewriter &rewriter) const override {
    Value lstmInput = lstmOp.input();
    StringRef prevLayer = lstmOp.prev_layer();
    if (!strcmp(prevLayer.data(), "bidir") ||
        !strcmp(prevLayer.data(), "uni")) {
      return failure();
    }

    // Search for zlow.lstm/gru op that generates the input argument.
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
      return failure();
    } else if (!strcmp(directionAttr.data(), "bidirectional")) {
      prevLayerAttr = rewriter.getStringAttr("bidir");
    } else {
      prevLayerAttr = rewriter.getStringAttr("uni");
    }
    // Update a zlow.lstm operation.
    lstmOp.prev_layerAttr(prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForLSTMOp(
        lstmOp.input_weights(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForLSTMOp(
        lstmOp.input_bias(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForLSTMOp(
        lstmOp.hidden_weights(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForLSTMOp(
        lstmOp.hidden_bias(), prevLayerAttr);

    return success();
  }
};

class SetPrevLayerInGRUOpPattern : public OpRewritePattern<ZLowGRUOp> {
public:
  using OpRewritePattern<ZLowGRUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZLowGRUOp gruOp, PatternRewriter &rewriter) const override {
    Value gruInput = gruOp.input();
    StringRef prevLayer = gruOp.prev_layer();
    if (!strcmp(prevLayer.data(), "bidir") ||
        !strcmp(prevLayer.data(), "uni")) {
      return failure();
    }

    // Search for zlow.lstm/gru op that generates the input argument.
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
      return failure();
    } else if (!strcmp(directionAttr.data(), "bidirectional")) {
      prevLayerAttr = rewriter.getStringAttr("bidir");
    } else {
      prevLayerAttr = rewriter.getStringAttr("uni");
    }
    // Update a zlow.gru operation.
    gruOp.prev_layerAttr(prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForGRUOp(
        gruOp.input_weights(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForGRUOp(
        gruOp.input_bias(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForGRUOp(
        gruOp.hidden_weights(), prevLayerAttr);
    setPrevLayerAttrOfGeneratorZLowStickForGRUOp(
        gruOp.hidden_bias(), prevLayerAttr);

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
    patterns.insert<StickViewUnstickRemovalPattern>(&getContext(), 2);
    patterns.insert<SetPrevLayerInLSTMOpPattern>(&getContext(), 1);
    patterns.insert<SetPrevLayerInGRUOpPattern>(&getContext(), 1);

    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> createZLowRewritePass() {
  return std::make_unique<ZLowRewritePass>();
}

} // namespace zlow
} // namespace onnx_mlir
