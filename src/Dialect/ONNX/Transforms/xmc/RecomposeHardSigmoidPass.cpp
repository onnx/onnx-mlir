// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/Support/Debug.h"

#include <cmath>

#define DEBUG_TYPE "recompose-hard-sigmoid"

using namespace mlir;

namespace {

static std::optional<float> getConstScalarF32(Value v) {
  if (!v || isa<NoneType>(v.getType()))
    return std::nullopt;
  auto cst = v.getDefiningOp<ONNXConstantOp>();
  if (!cst)
    return std::nullopt;
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(cst.getValueAttr());
  if (!elementsAttr)
    return std::nullopt;
  if (elementsAttr.isSplat()) {
    Type et = elementsAttr.getElementType();
    if (!isa<FloatType>(et))
      return std::nullopt;
    APFloat apf = elementsAttr.getSplatValue<APFloat>();
    return static_cast<float>(apf.convertToDouble());
  }
  auto shapedTy = dyn_cast<ShapedType>(elementsAttr.getType());
  if (!shapedTy || !shapedTy.hasStaticShape() || shapedTy.getNumElements() != 1)
    return std::nullopt;
  Attribute firstAttr = *elementsAttr.getValues<Attribute>().begin();
  if (auto f = dyn_cast<FloatAttr>(firstAttr))
    return static_cast<float>(f.getValueAsDouble());
  return std::nullopt;
}

struct RecomposeHardSigmoidPattern : public OpRewritePattern<ONNXClipOp> {
  using OpRewritePattern<ONNXClipOp>::OpRewritePattern;

  static constexpr float kCanonicalAlpha = 1.0f / 6.0f;
  static constexpr float kCanonicalBeta = 0.5f;
  static constexpr float kConstTol = 1e-2f;

  LogicalResult matchAndRewrite(
      ONNXClipOp clipOp, PatternRewriter &rewriter) const override {
    std::optional<float> clipMin = getConstScalarF32(clipOp.getMin());
    std::optional<float> clipMax = getConstScalarF32(clipOp.getMax());
    if (!clipMin || !clipMax || std::fabs(*clipMin) > kConstTol ||
        std::fabs(*clipMax - 1.0f) > kConstTol)
      return rewriter.notifyMatchFailure(
          clipOp, "Clip bounds are not approximately (0, 1)");

    auto addOp = clipOp.getInput().getDefiningOp<ONNXAddOp>();
    if (!addOp || !addOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          clipOp, "Clip input is not a single-use ONNXAddOp");

    Value mulVal = addOp.getA();
    Value betaVal = addOp.getB();
    auto mulOp = mulVal.getDefiningOp<ONNXMulOp>();
    if (!mulOp) {
      mulVal = addOp.getB();
      betaVal = addOp.getA();
      mulOp = mulVal.getDefiningOp<ONNXMulOp>();
      if (!mulOp)
        return rewriter.notifyMatchFailure(
            addOp, "neither Add operand is produced by an ONNXMulOp");
    }
    if (!mulOp->hasOneUse())
      return rewriter.notifyMatchFailure(mulOp, "Mul has more than one use");

    std::optional<float> beta = getConstScalarF32(betaVal);
    if (!beta)
      return rewriter.notifyMatchFailure(
          addOp, "Add's beta operand is not a constant scalar f32");

    Value xVal = mulOp.getA();
    std::optional<float> alpha = getConstScalarF32(mulOp.getB());
    if (!alpha) {
      xVal = mulOp.getB();
      alpha = getConstScalarF32(mulOp.getA());
      if (!alpha)
        return rewriter.notifyMatchFailure(
            mulOp, "Mul's alpha operand is not a constant scalar f32");
    }

    if (std::fabs(*alpha - kCanonicalAlpha) > kConstTol ||
        std::fabs(*beta - kCanonicalBeta) > kConstTol)
      return rewriter.notifyMatchFailure(mulOp,
          "Mul/Add constants are not canonical HardSigmoid (~1/6, ~0.5)");

    Location loc = mlir::FusedLoc::get(rewriter.getContext(),
        {mulOp.getLoc(), addOp.getLoc(), clipOp.getLoc()});

    auto hsigOp = rewriter.create<ONNXHardSigmoidOp>(loc,
        clipOp.getResult().getType(), xVal, rewriter.getF32FloatAttr(0.2f),
        rewriter.getF32FloatAttr(0.5f));

    rewriter.replaceOp(clipOp, hsigOp.getResult());
    if (addOp->use_empty())
      rewriter.eraseOp(addOp);
    if (mulOp->use_empty())
      rewriter.eraseOp(mulOp);

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct RecomposeHardSigmoidPass : public PassWrapper<RecomposeHardSigmoidPass,
                                      OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "recompose-hard-sigmoid"; }
  StringRef getDescription() const override {
    return "Fold Mul(x, ~1/6) -> Add(., ~0.5) -> Clip(., 0, 1) into "
           "onnx.HardSigmoid (alpha=0.2, beta=0.5).";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RecomposeHardSigmoidPattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createRecomposeHardSigmoidPass() {
  return std::make_unique<RecomposeHardSigmoidPass>();
}

} // namespace onnx_mlir
