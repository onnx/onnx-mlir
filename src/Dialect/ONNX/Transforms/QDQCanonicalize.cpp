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
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_QDQCANONICALIZEPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"

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
    : public impl::QDQCanonicalizePassBase<QDQCanonicalizePass> {
public:
  using Base::Base;
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
            GreedyRewriteConfig().setListener(&rnUpdater))))
      signalPassFailure();
  }

private:
  FrozenRewritePatternSet frozenPatterns;
};

} // namespace onnx_mlir
