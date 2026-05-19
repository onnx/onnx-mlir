// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
// AddRequantForOutputConvPass
//
// Mirrors xcompiler's
// xcompiler-src/src/pass/passes/AddRequantForOutputConvPass.cpp
//
// When a quantized-output op has multiple fanouts AND one fanout is the
// `quant.scast -> onnx.DequantizeLinear` "output edge" of the quantized
// region, REPLACE the scast with a placeholder XCOMPILERRequantize that
// produces the same storage type so the DQ has its own dedicated
// requantize handle for downstream passes. This mirrors the shape used by
// ConvertScastAndDQToRequantizePattern in ConvertSCastPairToRequantizePass.
//
// Real post-QuantTypes IR shape:
//   producer (!quant.uniform[s,zp])
//     |---> quant.scast (!quant.uniform -> ui8) -> DQ -> ... f32 output
//     |---> (other quantized consumers ...)
//
// After:
//   producer (!quant.uniform[s,zp])
//     |---> XCOMPILERRequantize(a_scale=[1.0],a_zp=[0];
//     |                        y_scale=[1.0],y_zp=[0]) -> ui8
//     |       \---> DQ -> ... f32 output
//     |---> (other quantized consumers, unchanged)
//
// The a/y placeholder values (scale=1.0, zp=0) mirror xcompiler's flow
// exactly -- they are intentionally NOT derived from the producer's quant
// type. Downstream passes overwrite them with the real per-output
// requantize parameters. The new XCOMPILERRequantize absorbs the scast's
// type-unwrap role while providing the explicit requantize hook the
// xcompiler pass was designed to introduce.
//
// Producer allow-list (verified mapping of xcompiler's
// {qlinear-conv2d, qlinear-eltwise}):
//   { XFEConvOp, XCOMPILERDepthwiseConvOp, XCOMPILERFusedEltwiseOp }
//
// xcompiler gates that are dropped here: is_4x4_cmc_overlay, and the
// qlinear-l2_normalize producer (no xmc representation today).
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

//===----------------------------------------------------------------------===//
// Pattern: replace the producer->scast edge with a placeholder
// XCOMPILERRequantize when the producer has multiple fanouts and feeds
// DequantizeLinear. Placeholder attributes (a/y scale = 1.0, a/y zp = 0)
// mirror xcompiler's flow; downstream passes will fill in the real
// requantize parameters later.
//===----------------------------------------------------------------------===//

/// Anchor on `ONNXDequantizeLinearOp` (the "output edge" of the quantized
/// region) and look through `quant::StorageCastOp` to find the producer.
/// Fires only when the producer is one of the verified xcompiler-mapped
/// quantized compute ops AND its quant-typed result has multiple uses
/// (matches xcompiler's `!if_single_fanout(qconv)`). The scast is then
/// replaced by a placeholder XCOMPILERRequantize that produces the same
/// storage type, so the DQ ends up consuming the requantize directly.
struct AddRequantForOutputConvPattern
    : public OpRewritePattern<ONNXDequantizeLinearOp> {
  using OpRewritePattern<ONNXDequantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXDequantizeLinearOp dqOp, PatternRewriter &rewriter) const override {
    // 1. Look through quant.scast: !quant.uniform -> storage type -> DQ.
    auto scast = dqOp.getX().getDefiningOp<quant::StorageCastOp>();
    if (!scast)
      return failure();

    Value producerQuantVal = scast.getOperand();

    // 2. Producer allow-list (verified mapping of xcompiler's
    //    {qlinear-conv2d, qlinear-eltwise}).
    Operation *producer = producerQuantVal.getDefiningOp();
    if (!producer ||
        !isa<XFEConvOp, XCOMPILERDepthwiseConvOp, XCOMPILERFusedEltwiseOp>(
            producer))
      return failure();

    // 3. Multi-fanout check on the producer's QUANT result (matches
    //    xcompiler AddRequantForOutputConvPass: !internal::if_single_fanout).
    if (producerQuantVal.hasOneUse())
      return failure();

    // 4. Element-type check: must be a uniform-quantized tensor type. We
    //    only fire when the producer's result is genuinely quantized.
    auto rtt = dyn_cast<RankedTensorType>(producerQuantVal.getType());
    if (!rtt)
      return failure();
    Type elt = rtt.getElementType();
    if (!isa<quant::UniformQuantizedType, quant::UniformQuantizedPerAxisType>(
            elt))
      return failure();

    // 5. Build placeholder a_*/y_* attrs identical to xcompiler's flow:
    //    a_scale=[1.0], a_zero_point=[0], y_scale=[1.0], y_zero_point=[0].
    //    These are intentionally NOT derived from the producer's quant
    //    type; downstream passes overwrite them with the real requantize
    //    parameters. The XCOMPILERRequantize verifier only requires
    //    a_scale/a_zp and y_scale/y_zp shape consistency, not consistency
    //    with the input/output tensor quant types, so size-1 placeholders
    //    are legal regardless of whether the producer is per-tensor or
    //    per-axis quantized.
    ArrayAttr aScale =
        rewriter.getArrayAttr({rewriter.getF32FloatAttr(1.0f)});
    ArrayAttr aZp = rewriter.getI64ArrayAttr({0});
    ArrayAttr yScale = aScale;
    ArrayAttr yZp = aZp;

    // 6. Create the placeholder XCOMPILERRequantizeOp with the SCAST's
    //    result type (storage type, e.g. ui8) so it can replace the scast
    //    wholesale. Propagate ResultNames via the canonical helper, then
    //    replace the scast with the requantize's result (other uses of the
    //    producer's quant-typed value are untouched).
    rewriter.setInsertionPoint(scast);
    auto rq = rewriter.create<XCOMPILERRequantizeOp>(producer->getLoc(),
        scast.getResult().getType(), producerQuantVal, aScale, aZp, yScale,
        yZp);

    // ResultNamesUpdater::notifyOperationReplaced(producer, rq) is the
    // canonical helper that copies ResultNames AND runs inferTensorNames on
    // the new op's operands. The greedy-rewriter listener machinery only
    // auto-invokes it on op REPLACEMENT of the LISTENED rewrite; here we
    // call it explicitly so the new Requantize inherits the producer's
    // ResultNames in the same way a true replacement would.
    onnx_mlir::ResultNamesUpdater rnUpdater;
    rnUpdater.notifyOperationReplaced(producer, rq.getOperation());

    // Replace the scast with the requantize result. The DQ (and any other
    // user of the scast's storage-typed result, though typically there is
    // none) now consumes the requantize directly.
    rewriter.replaceOp(scast, rq.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

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

    // ResultNamesUpdater listener is intentionally NOT attached. Our
    // pattern manually invokes notifyOperationReplaced(producer, rq) to
    // copy ResultNames from the *producer* (not the scast) onto the new
    // Requantize. If the listener were attached, the subsequent
    // replaceOp(scast, rq.getResult()) would re-fire the listener and
    // overwrite the producer-sourced ResultNames with the scast's (or
    // unset them), which is not what we want.
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
