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
///   zlow.lstm(%input, %h0, %c0, %input_weights, %input_bias,
///             %hidden_weights, %hidden_bias, %work_area, %shape,
///             %hn_output, %cf_output,
///             %direction, %return_all_steps, %prev_layer)
///   ...
///   zlow.lstm(%hn_output, %h0_, %c0_, %input_weights_, %input_bias_,
///             %hidden_weights_, %hidden_bias_, %work_area_, %shape_,
///             %hn_output_, %cf_output_,
///             %direction_, %return_all_steps_, "none")
/// ===>
///   zlow.lstm(%input, %h0, %c0, %input_weights, %input_bias,
///             %hidden_weights, %hidden_bias, %work_area, %shape,
///             %hn_output, %cf_output,
///             %direction, %return_all_steps, %prev_layer)
///   ...
///   zlow.lstm(%hn_output, %h0_, %c0_, %input_weights_, %input_bias_,
///             %hidden_weights_, %hidden_bias_, %work_area_, %shape_,
///             %hn_output_, %cf_output_,
///             %direction_, %return_all_steps_,
///             (%direction == "bidirectional) ? "bidir" : "uni") #was "none"
///
/// ```
/// by changing the prev_layer parameter from "none" to "bidir" or "uni"
/// according to the operation of the input argument.
///   "none" if input tensor is not from a previous RNN layer
///   "uni" if input tensor is uni-directional output from a previous RNN layer
///   "bidir" if input tensor is bi-directional output from a previous RNN layer
//
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
      char *opName = (char *)user->getName().getStringRef().data();
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
      char *opName = (char *)user->getName().getStringRef().data();
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
    : public PassWrapper<ZLowRewriteFinalPass, OperationPass<FuncOp>> {
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
