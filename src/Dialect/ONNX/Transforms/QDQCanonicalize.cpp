// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <memory>

#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

using namespace mlir;

namespace onnx_mlir {

struct FoldQDQPattern : public OpRewritePattern<ONNXQuantizeLinearOp> {
  FoldQDQPattern(MLIRContext *context, int64_t maxRoundTripDiff = 0)
      : OpRewritePattern<ONNXQuantizeLinearOp>(context),
        maxRoundTripDiff(maxRoundTripDiff) {}

  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {

    auto dqOp = qOp.getX().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp)
      return failure();
    if (!isDequantQuantSame(dqOp, qOp, maxRoundTripDiff))
      return failure();
    rewriter.replaceOp(qOp, dqOp.getX());
    return success();
  }

private:
  // Maximum allowed per-code difference for the DQ->Q round-trip. 0 requires an
  // exact (bit-for-bit scale/zero-point) match; larger values allow folding
  // when the round-trip over the full storage range stays within this many
  // codes (per-tensor scalar params only).
  int64_t maxRoundTripDiff;
};

void getDQBinaryQPatterns(RewritePatternSet &patterns, MLIRContext *context);

void getRemoveQDQAroundOpPatterns(
    RewritePatternSet &patterns, MLIRContext *context);

class QDQCanonicalizePass
    : public PassWrapper<QDQCanonicalizePass, OperationPass<func::FuncOp>> {
public:
  Option<bool> removeBinary{*this, "remove-binary", llvm::cl::init(false)};
  Option<bool> removeQDQAroundOps{
      *this, "remove-qdq-around-ops", llvm::cl::init(false)};
  Option<int64_t> maxRoundTripDiff{*this, "max-round-trip-diff",
      llvm::cl::desc("Max per-code DQ->Q round-trip difference tolerated when "
                     "folding a redundant QDQ pair (0 = exact match)"),
      llvm::cl::init(0)};

  StringRef getArgument() const override { return "qdq-canonicalize"; }

  QDQCanonicalizePass(
      bool removeBinary, bool removeQDQAroundOps, int64_t maxRoundTripDiff) {
    this->removeBinary = removeBinary;
    this->removeQDQAroundOps = removeQDQAroundOps;
    this->maxRoundTripDiff = maxRoundTripDiff;
  }

  QDQCanonicalizePass(const QDQCanonicalizePass &pass)
      : frozenPatterns(pass.frozenPatterns) {
    copyOptionValuesFrom(&pass);
  }

  LogicalResult initialize(MLIRContext *context) override {
    mlir::RewritePatternSet patterns(context);
    if (removeBinary)
      getDQBinaryQPatterns(patterns, context);
    if (removeQDQAroundOps)
      getRemoveQDQAroundOpPatterns(patterns, context);
    patterns.add<FoldQDQPattern>(context, maxRoundTripDiff);
    frozenPatterns = std::move(patterns);
    return success();
  }

  void runOnOperation() override {
    onnx_mlir::ResultNamesUpdater rnUpdater;
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns,
            GreedyRewriteConfig{.listener = &rnUpdater})))
      signalPassFailure();
  }

private:
  FrozenRewritePatternSet frozenPatterns;
};

std::unique_ptr<mlir::Pass> createQDQCanonicalizePass(
    bool removeBinary, bool removeQDQAroundOps, int64_t maxRoundTripDiff) {
  return std::make_unique<QDQCanonicalizePass>(
      removeBinary, removeQDQAroundOps, maxRoundTripDiff);
}

} // namespace onnx_mlir
