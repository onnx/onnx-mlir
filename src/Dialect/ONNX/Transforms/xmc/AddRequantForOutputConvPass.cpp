// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// AddRequantForOutputConvPass: on a quantized compute op with multi-fanout
// where one fanout is `quant.scast -> DequantizeLinear`, replace the scast
// with a placeholder XCOMPILERRequantize on that output edge.

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

using namespace mlir;

namespace {

struct AddRequantForOutputConvPattern
    : public OpRewritePattern<ONNXDequantizeLinearOp> {
  using OpRewritePattern<ONNXDequantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXDequantizeLinearOp dqOp, PatternRewriter &rewriter) const override {
    auto scast = dqOp.getX().getDefiningOp<quant::StorageCastOp>();
    if (!scast)
      return failure();

    Value producerQuantVal = scast.getOperand();

    Operation *producer = producerQuantVal.getDefiningOp();
    if (!producer ||
        !isa<XFEConvOp, XCOMPILERDepthwiseConvOp, XCOMPILERFusedEltwiseOp>(
            producer))
      return failure();

    if (producerQuantVal.hasOneUse())
      return failure();

    auto rtt = dyn_cast<RankedTensorType>(producerQuantVal.getType());
    if (!rtt)
      return failure();
    if (!isa<quant::UniformQuantizedType, quant::UniformQuantizedPerAxisType>(
            rtt.getElementType()))
      return failure();

    // Placeholder a/y attrs (scale=1.0, zp=0); downstream passes overwrite
    // them with the real requantize params. The XCOMPILERRequantize
    // verifier only checks a/y mutual shape consistency, so size-1 attrs
    // are legal for both per-tensor and per-axis producers.
    ArrayAttr aScale = rewriter.getArrayAttr({rewriter.getF32FloatAttr(1.0f)});
    ArrayAttr aZp = rewriter.getI64ArrayAttr({0});
    ArrayAttr yScale = aScale;
    ArrayAttr yZp = aZp;

    rewriter.setInsertionPoint(scast);
    auto rq = rewriter.create<XCOMPILERRequantizeOp>(producer->getLoc(),
        scast.getResult().getType(), producerQuantVal, aScale, aZp, yScale,
        yZp);

    // Explicit notify: the greedy-rewriter listener only fires on op
    // replacement, but here we INSERT a new op then modify-in-place, so we
    // call the helper ourselves to copy ResultNames from the producer (not
    // the scast) onto the new Requantize.
    onnx_mlir::ResultNamesUpdater rnUpdater;
    rnUpdater.notifyOperationReplaced(producer, rq.getOperation());

    rewriter.replaceOp(scast, rq.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct AddRequantForOutputConvPass
    : public PassWrapper<AddRequantForOutputConvPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "add-requant-for-output-conv";
  }
  StringRef getDescription() const override {
    return "Insert placeholder XCOMPILERRequantize on quantized-op -> "
           "DequantizeLinear edges where the producer has multiple fanouts.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<quant::QuantDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<AddRequantForOutputConvPattern>(context);

    // ResultNamesUpdater listener is NOT attached: the pattern's
    // replaceOp(scast, ...) would fire it and overwrite the
    // producer-sourced ResultNames that the pattern just copied onto the
    // new Requantize.
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createAddRequantForOutputConvPass() {
  return std::make_unique<AddRequantForOutputConvPass>();
}

} // namespace onnx_mlir
