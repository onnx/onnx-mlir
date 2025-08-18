//===- QDQOpt.cpp - Remove QDQ operations --------*- C++ -*-===//
//
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

#include <cmath>

using namespace mlir;
using namespace onnx_mlir;

namespace {

//===----------------------------------------------------------------------===//
// Pattern to remove QDQ pairs
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Pass to run QDQ removal
//===----------------------------------------------------------------------===//

struct QDQOptONNXToONNXPass
    : public PassWrapper<QDQOptONNXToONNXPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QDQOptONNXToONNXPass)
  StringRef getArgument() const override { return "dqq-opt-onnx-to-onnx"; }
  StringRef getDescription() const override {
    return "Remove DqQ ops and surrounding DqQ if safe.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldQDQPattern>(&getContext());
    if (failed(applyPatternsGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createQDQOptONNXToONNXPass() {
  return std::make_unique<QDQOptONNXToONNXPass>();
}
} // namespace onnx_mlir