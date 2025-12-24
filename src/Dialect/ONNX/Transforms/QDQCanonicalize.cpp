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

using namespace mlir;

namespace onnx_mlir {

struct FoldQDQPattern : public OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {

    auto dqOp = qOp.getX().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp)
      return failure();
    if (!isDequantQuantSame(dqOp, qOp))
      return failure();
    rewriter.replaceOp(qOp, dqOp.getX());
    return success();
  }
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

  StringRef getArgument() const override { return "qdq-canonicalize"; }

  QDQCanonicalizePass(bool removeBinary, bool removeQDQAroundOps) {
    this->removeBinary = removeBinary;
    this->removeQDQAroundOps = removeQDQAroundOps;
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
    patterns.add<FoldQDQPattern>(context);
    frozenPatterns = std::move(patterns);
    return success();
  }

  void runOnOperation() override {
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns)))
      signalPassFailure();
  }

private:
  FrozenRewritePatternSet frozenPatterns;
};

std::unique_ptr<mlir::Pass> createQDQCanonicalizePass(
    bool removeBinary, bool removeQDQAroundOps) {
  return std::make_unique<QDQCanonicalizePass>(
      removeBinary, removeQDQAroundOps);
}

} // namespace onnx_mlir
