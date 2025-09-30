/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- DonotUseAttentionMask.cpp - Remove masking in Attention layer
//----===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {} // namespace onnx_mlir

namespace {

// Attention masking is done via AddOp.
// This is the pattern of masking:
// x1 = MatMul()
// x2 = Add(x1, mask)
// x3 = Softmax(x2)
//
// where, mask is computed indirectly from the attention_mask input of the
// model.
//
// This pattern will match the AddOp sandwitched between MatMul and Softmax,
// and that the mask input comes from the attention_mask input.
//
// A replaced pattern is:
// x1 = MatMul()
// x3 = Softmax(x1)
struct RemoveAttentionMaskPattern : public OpRewritePattern<ONNXAddOp> {
  RemoveAttentionMaskPattern(MLIRContext *context, Value attentionMaskArg)
      : OpRewritePattern(context), attentionMaskArg(attentionMaskArg) {}
  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    Value A = addOp.getA();
    Value B = addOp.getB();

    // Match MatMul and get mask value.
    Value mask, mmOutput;
    if (A.getDefiningOp<ONNXMatMulOp>()) {
      mask = B;
      mmOutput = A;
    } else if (B.getDefiningOp<ONNXMatMulOp>()) {
      mask = A;
      mmOutput = B;
    } else
      return failure();

    // Match softmax.
    ONNXSoftmaxOp softmaxOp;
    for (Operation *user : addOp.getC().getUsers()) {
      if (auto sop = dyn_cast<ONNXSoftmaxOp>(user)) {
        softmaxOp = sop;
        break;
      }
    }
    if (!softmaxOp)
      return failure();

    // Check that mask comes from attention_mask.

    // Rewrite: bypass the AddOp.
    rewriter.modifyOpInPlace(
        softmaxOp, [&]() { softmaxOp.setOperand(mmOutput); });
    return success();
  }

private:
  Value attentionMaskArg;
};

} // namespace

namespace {

struct DonotUseAttentionMaskPass : public PassWrapper<DonotUseAttentionMaskPass,
                                       OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DonotUseAttentionMaskPass)

  DonotUseAttentionMaskPass() = default;
  DonotUseAttentionMaskPass(const DonotUseAttentionMaskPass &pass)
      : mlir::PassWrapper<DonotUseAttentionMaskPass,
            OperationPass<func::FuncOp>>() {}
  DonotUseAttentionMaskPass(uint64_t argIdx) { this->argIdx = argIdx; };

  StringRef getArgument() const override { return "do-not-use-attention-mask"; }

  StringRef getDescription() const override {
    return "Do not use attention_mask in a self_attention layer";
  }

  // Usage: onnx-mlir-opt --do-not-use-attention-mask='arg-idx=1'
  Option<bool> argIdx{*this, "arg-idx",
      llvm::cl::desc("Argument index of attention_mask in the function"),
      ::llvm::cl::init(1)};

  void runOnOperation() final;
};

void DonotUseAttentionMaskPass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  // ConversionTarget target(getContext());
  RewritePatternSet patterns(context);

  Value attentionMask = function.getArgument(this->argIdx);
  patterns.insert<RemoveAttentionMaskPattern>(context, attentionMask);

  GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);

  if (failed(applyPatternsGreedily(function, std::move(patterns), config)))
    signalPassFailure();
}

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createDonotUseAttentionMaskPass(
    uint64_t argIdx) {
  return std::make_unique<DonotUseAttentionMaskPass>(argIdx);
}
