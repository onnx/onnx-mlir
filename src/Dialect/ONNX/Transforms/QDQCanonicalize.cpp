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
  // Controls how closely the DQ->Q pair must act as an identity. For every
  // quantized integer x in the storage range, DequantizeLinear(x) followed
  // by QuantizeLinear must produce a value no further than this from x
  // (e.g. maxRoundTripDiff=2 allows x=1000 to come back as 999..1002).
  // 0 requires bit-for-bit identical scale and zero-point; a small positive
  // value (e.g. 8) tolerates tiny floating-point scale differences.
  // Only applies to per-tensor (scalar) quantization params.
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
      llvm::cl::desc("Maximum absolute difference allowed between an input "
                     "integer and its DQ->Q output, checked over the full "
                     "storage range. 0 requires bit-exact scale and "
                     "zero-point; >0 tolerates near-equal parameters."),
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
