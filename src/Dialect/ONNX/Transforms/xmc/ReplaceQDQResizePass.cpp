// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include <cmath>

#define DEBUG_TYPE "replace-qdq-resize"

using namespace mlir;

namespace {

static bool hasMatchingUniformQuant(
    quant::UniformQuantizedType typeA, quant::UniformQuantizedType typeB) {
  return typeA.getStorageType() == typeB.getStorageType() &&
         std::fabs(typeA.getScale() - typeB.getScale()) <= 1e-6 &&
         typeA.getZeroPoint() == typeB.getZeroPoint();
}

struct ReplaceQDQResizeToAddPattern
    : public OpRewritePattern<XFEResizeOp> {
  using OpRewritePattern<XFEResizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      XFEResizeOp resizeOp, PatternRewriter &rewriter) const override {
    if (!resizeOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          resizeOp, "resize must have a single user");

    Value input = resizeOp.getX();
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(resizeOp.getType());
    if (!inType || !outType)
      return rewriter.notifyMatchFailure(
          resizeOp, "input and result must be ranked tensor types");

    if (inType.getRank() != 4 || outType.getRank() != 4)
      return rewriter.notifyMatchFailure(
          resizeOp, "only rank-4 NHWC resize is supported");

    if (!inType.hasStaticShape() || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          resizeOp, "input and output shapes must be static");

    ArrayRef<int64_t> inShape = inType.getShape();
    ArrayRef<int64_t> outShape = outType.getShape();
    if (inShape[1] != 1 || inShape[2] != 1)
      return rewriter.notifyMatchFailure(
          resizeOp, "input spatial dims (H, W) must both be 1");
    if (outShape[0] != inShape[0] || outShape[3] != inShape[3])
      return rewriter.notifyMatchFailure(
          resizeOp, "batch and channel dims must match between input and output");
    if (outShape[1] <= 1 && outShape[2] <= 1)
      return rewriter.notifyMatchFailure(
          resizeOp, "output spatial dims must be larger than 1");

    auto inQuant =
        dyn_cast<quant::UniformQuantizedType>(inType.getElementType());
    auto outQuant =
        dyn_cast<quant::UniformQuantizedType>(outType.getElementType());
    if (!inQuant || !outQuant)
      return rewriter.notifyMatchFailure(
          resizeOp, "input and output must be uniformly quantized");

    if (!hasMatchingUniformQuant(inQuant, outQuant))
      return rewriter.notifyMatchFailure(resizeOp,
          "input and output quant types must have matching scale and zero_point");

    Location loc = resizeOp.getLoc();

    int64_t zeroPoint = outQuant.getZeroPoint();
    Type storageTy = outQuant.getStorageType();

    auto zpStorageType = RankedTensorType::get(outShape, storageTy);
    auto zpSplatAttr = DenseElementsAttr::get(
        zpStorageType, rewriter.getIntegerAttr(storageTy, zeroPoint));
    auto valueNamedAttr = rewriter.getNamedAttr("value", zpSplatAttr);
    auto zpTensorConst = rewriter.create<ONNXConstantOp>(loc, outType,
        ValueRange{}, ArrayRef<NamedAttribute>{valueNamedAttr});

    auto addOp = rewriter.create<ONNXAddOp>(
        loc, outType, input, zpTensorConst.getResult());

    onnx_mlir::ResultNamesUpdater().notifyOperationReplaced(resizeOp, addOp);
    rewriter.replaceOp(resizeOp, addOp.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ReplaceQDQResizePass
    : public PassWrapper<ReplaceQDQResizePass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "replace-qdq-resize"; }
  StringRef getDescription() const override {
    return "Replace quantized XFEResize with 1x1 spatial input by a "
           "broadcasting onnx.Add against a splat zero_point constant of the "
           "output shape (matches xcompiler ReplaceQDQResizePass IPU path).";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReplaceQDQResizeToAddPattern>(context);

    GreedyRewriteConfig config;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config))) {
      getOperation().emitError(
          "replace-qdq-resize: greedy pattern rewrite did not converge");
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createReplaceQDQResizePass() {
  return std::make_unique<ReplaceQDQResizePass>();
}

} // namespace onnx_mlir
