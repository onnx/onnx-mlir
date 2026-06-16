// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// RemoveNoOpRequantizePass: drop an XCOMPILERRequantize whose output scale and
// zero-point (from the result quant type) match its input's, so the requantize
// is a no-op and its consumer can read the producer directly.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include <cmath>

using namespace mlir;

namespace {

static quant::UniformQuantizedType perTensorQuant(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  if (!rt)
    return nullptr;
  return dyn_cast<quant::UniformQuantizedType>(rt.getElementType());
}

struct RemoveNoOpRequantizePattern
    : public OpRewritePattern<XCOMPILERRequantizeOp> {
  using OpRewritePattern<XCOMPILERRequantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      XCOMPILERRequantizeOp reqOp, PatternRewriter &rewriter) const override {
    Value x = reqOp.getX();
    auto inQ = perTensorQuant(x.getType());
    auto outQ = perTensorQuant(reqOp.getResult().getType());
    if (!inQ || !outQ)
      return failure();

    // Storage dtype must be unchanged (else it is a real requantize).
    if (inQ.getStorageType() != outQ.getStorageType())
      return failure();

    // Producer must have a single fan-out.
    if (!x.hasOneUse())
      return failure();

    // Remove only when output scale/zero-point match the input (no-op).
    if (std::abs(inQ.getScale() - outQ.getScale()) >= 1e-6 ||
        inQ.getZeroPoint() != outQ.getZeroPoint())
      return failure();

    rewriter.replaceOp(reqOp, x);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct RemoveNoOpRequantizePass : public PassWrapper<RemoveNoOpRequantizePass,
                                      OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveNoOpRequantizePass)

  StringRef getArgument() const override { return "remove-noop-requantize"; }
  StringRef getDescription() const override {
    return "Remove a no-op XCOMPILERRequantize (input quant == output quant).";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<quant::QuantDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RemoveNoOpRequantizePattern>(context);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createRemoveNoOpRequantizePass() {
  return std::make_unique<RemoveNoOpRequantizePass>();
}

} // namespace onnx_mlir
