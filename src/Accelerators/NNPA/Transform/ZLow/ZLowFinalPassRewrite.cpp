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
      const char *opName = user->getName().getStringRef().data();
      if (!strcmp(opName, "zlow.lstm")) {
        ZLowLSTMOp userLstmOp = llvm::dyn_cast<ZLowLSTMOp>(user);
        if (userLstmOp != lstmOp) {
          directionAttr = userLstmOp.direction();
          break;
        }
      }
      if (!strcmp(opName, "zlow.gru")) {
        directionAttr = llvm::dyn_cast<ZLowGRUOp>(user).direction();
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
    // Emit a ZLow operation.
    rewriter.create<ZLowLSTMOp>(lstmOp.getLoc(), lstmOp.input(), lstmOp.h0(),
        lstmOp.c0(), lstmOp.input_weights(), lstmOp.input_bias(),
        lstmOp.hidden_weights(), lstmOp.hidden_bias(), lstmOp.work_area(),
        lstmOp.shape(), lstmOp.hn_output(), lstmOp.cf_output(),
        lstmOp.directionAttr(), lstmOp.return_all_stepsAttr(), prevLayerAttr);
    rewriter.eraseOp(lstmOp);

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
      const char *opName = user->getName().getStringRef().data();
      if (!strcmp(opName, "zlow.gru")) {
        ZLowGRUOp userGruOp = llvm::dyn_cast<ZLowGRUOp>(user);
        if (userGruOp != gruOp) {
          directionAttr = userGruOp.direction();
          break;
        }
      }
      if (!strcmp(opName, "zlow.lstm")) {
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
    // Emit a ZLow operation.
    rewriter.create<ZLowGRUOp>(gruOp.getLoc(), gruOp.input(), gruOp.h0(),
        gruOp.input_weights(), gruOp.input_bias(), gruOp.hidden_weights(),
        gruOp.hidden_bias(), gruOp.work_area(), gruOp.shape(),
        gruOp.hn_output(), gruOp.directionAttr(), gruOp.return_all_stepsAttr(),
        prevLayerAttr);
    rewriter.eraseOp(gruOp);

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
