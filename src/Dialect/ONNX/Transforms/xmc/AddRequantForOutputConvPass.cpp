// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
// AddRequantForOutputConvPass
//
// Ports xcompiler's AddRequantForOutputConvPass. For a producer in
//   { XFEConvOp, XCOMPILERDepthwiseConvOp, XCOMPILERFusedEltwiseOp }
// whose `!quant.uniform` result has multiple fanouts and one fanout is
// `quant.scast -> ONNXDequantizeLinearOp` (the "output edge"), replace the
// scast with a placeholder XCOMPILERRequantize that produces the same
// storage type. Placeholder attrs `a_scale=[1.0], a_zp=[0], y_scale=[1.0],
// y_zp=[0]` mirror xcompiler's flow exactly; downstream passes overwrite
// them with the real per-output requantize parameters.
//
// xcompiler gates dropped: is_4x4_cmc_overlay; qlinear-l2_normalize
// producer (no xmc representation today).
//===----------------------------------------------------------------------===//

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

    // Producer allow-list mirrors xcompiler's {qlinear-conv2d,
    // qlinear-eltwise}.
    Operation *producer = producerQuantVal.getDefiningOp();
    if (!producer ||
        !isa<XFEConvOp, XCOMPILERDepthwiseConvOp, XCOMPILERFusedEltwiseOp>(
            producer))
      return failure();

    // Mirrors xcompiler's `!internal::if_single_fanout(qconv)`.
    if (producerQuantVal.hasOneUse())
      return failure();

    auto rtt = dyn_cast<RankedTensorType>(producerQuantVal.getType());
    if (!rtt)
      return failure();
    if (!isa<quant::UniformQuantizedType, quant::UniformQuantizedPerAxisType>(
            rtt.getElementType()))
      return failure();

    // Placeholder a/y attrs (scale=1.0, zp=0) match xcompiler exactly --
    // downstream passes overwrite them with the real requantize params.
    // The XCOMPILERRequantize verifier only checks a/y mutual shape
    // consistency, so size-1 attrs are legal for per-tensor and per-axis
    // producers alike.
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
    return "Insert no-op XCOMPILERRequantize on quantized-op -> "
           "DequantizeLinear edges where the producer has multiple fanouts "
           "(mirrors xcompiler AddRequantForOutputConvPass).";
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
