/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZLowFinalRewrite.cpp - ZLow Rewrite Patterns ----------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This pass investigates operation that generate the input, and set the
// prev_layer attribute of zlow.lstm operation as follows.
///   "none" if input tensor is not from a previous RNN layer
///   "uni" if input tensor is uni-directional output from a previous RNN layer
///   "bidir" if input tensor is bi-directional output from a previous RNN layer
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

class SetPrevLayerInLSTMOpPattern : public OpRewritePattern<ZLowLSTMOp> {
public:
  using OpRewritePattern<ZLowLSTMOp>::OpRewritePattern;

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
 *  This Pass should be the final pass toupdate ZLowIR.
 */
class ZLowRewriteFinalPass
    : public PassWrapper<ZLowRewriteFinalPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "zlow-rewrite-final"; }

  StringRef getDescription() const override {
    return "Rewrite ZLow Ops. at final";
  }

  void runOnOperation() override {
    Operation *function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<SetPrevLayerInLSTMOpPattern>(&getContext());
    patterns.insert<SetPrevLayerInGRUOpPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> createZLowRewriteFinalPass() {
  return std::make_unique<ZLowRewriteFinalPass>();
}

} // namespace zlow
} // namespace onnx_mlir
