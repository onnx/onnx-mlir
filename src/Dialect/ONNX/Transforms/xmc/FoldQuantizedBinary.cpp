// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// XMC variant of FoldQuantizedBinary. Folds only quantized Div with a
// scalar constant RHS, matching xcompiler's
// TransferScalarConstInputDivToRequantizePass. Add/Sub/Mul are intentionally
// not folded here.

#include <cmath>
#include <llvm/ADT/APFloat.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <optional>

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

using namespace mlir;

namespace {

std::optional<double> getConstant(Value rhs) {
  auto constType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!constType)
    return {};

  auto constOp = rhs.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return {};

  auto attr = dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
  if (!attr || !attr.isSplat())
    return {};

  auto attrDtype = attr.getElementType();
  auto resultDtype = constType.getElementType();
  if (auto qType = dyn_cast<quant::UniformQuantizedType>(resultDtype);
      qType && qType.getStorageType() == attrDtype) {
    int64_t constVal;
    if (attrDtype.isUnsignedInteger())
      constVal = attr.getSplatValue<APInt>().getZExtValue();
    else
      constVal = attr.getSplatValue<APInt>().getSExtValue();
    return static_cast<double>(constVal - qType.getZeroPoint()) *
           qType.getScale();
  } else if (isa<FloatType>(resultDtype) && resultDtype == attrDtype) {
    return attr.getSplatValue<APFloat>().convertToDouble();
  }

  return {};
}

double convertToExpressedType(double value, quant::UniformQuantizedType qType) {
  if (auto fltType = dyn_cast<FloatType>(qType.getExpressedType())) {
    APFloat ap(value);
    bool losesInfo;
    ap.convert(
        fltType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
    value = ap.convertToDouble();
  }
  return value;
}

quant::UniformQuantizedType getDivOutQuantType(
    quant::UniformQuantizedType inQType, double binConst, Location loc) {
  if (binConst == 0.0)
    return nullptr;
  double newScale = convertToExpressedType(inQType.getScale() / binConst, inQType);
  return quant::UniformQuantizedType::getChecked(
      [&loc]() { return emitWarning(loc); }, inQType.getFlags(),
      inQType.getStorageType(), inQType.getExpressedType(), newScale,
      inQType.getZeroPoint(), inQType.getStorageTypeMin(),
      inQType.getStorageTypeMax());
}

} // namespace

namespace onnx_mlir {

class XmcFoldQuantizedDiv : public OpRewritePattern<ONNXDivOp> {
public:
  using OpRewritePattern<ONNXDivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXDivOp divOp, PatternRewriter &rewriter) const override {
    auto lhs = divOp->getOperand(0);
    auto rhs = divOp->getOperand(1);
    auto out = divOp->getResult(0);

    // No need of swapping inputs if first input is constant
    // Canonicalization takes care of that
    if (lhs.getDefiningOp<ONNXConstantOp>())
      return rewriter.notifyMatchFailure(divOp, "LHS should not be a constant");

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto outType = dyn_cast<RankedTensorType>(out.getType());
    if (!lhsType || !outType)
      return rewriter.notifyMatchFailure(divOp, "Not Ranked TensorTypes");

    auto lhsQType =
        dyn_cast<quant::UniformQuantizedType>(lhsType.getElementType());
    auto outQType =
        dyn_cast<quant::UniformQuantizedType>(outType.getElementType());
    if (!lhsQType || !outQType)
      return rewriter.notifyMatchFailure(
          divOp, "Not Quantized input and output types");

    double binConst;
    if (auto constOpt = getConstant(rhs))
      binConst = *constOpt;
    else
      return rewriter.notifyMatchFailure(divOp, "RHS not scalar constant");

    auto isONNXOp = [](Operation *op) -> bool {
      if (isa<ONNXDequantizeLinearOp, ONNXQuantizeLinearOp>(op))
        return false;
      return isa<ONNXDialect>(op->getDialect());
    };

    // Mirror xcompiler: only ever rewrite the Div's OUTPUT into a
    // requantize. Never modify the upstream LHS quant type annotation.
    if (!llvm::all_of(out.getUsers(), isONNXOp))
      return rewriter.notifyMatchFailure(
          divOp, "Output has non-ONNX users; cannot fold");

    Location binLoc = divOp->getLoc();
    auto newQType = getDivOutQuantType(lhsQType, binConst, binLoc);
    if (!newQType)
      return rewriter.notifyMatchFailure(divOp, "Cannot get new QType");

    auto scast = rewriter.create<quant::StorageCastOp>(
        binLoc, lhsType.clone(lhsQType.getStorageType()), lhs);
    auto replOp = rewriter.create<quant::StorageCastOp>(
        binLoc, outType.clone(newQType), scast);
    rewriter.replaceOp(divOp, replOp);
    // Since we fold DQ -> Div -> Q -> DQ into DQ, we should not be
    // propagating the ResultNames of Q
    replOp->removeAttr("ResultNames");

    return success();
  }
};

class XmcFoldQuantizedBinary
    : public PassWrapper<XmcFoldQuantizedBinary, OperationPass<func::FuncOp>> {
public:
  [[nodiscard]] StringRef getArgument() const override {
    return "xmc-fold-quantized-binary";
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<XmcFoldQuantizedDiv>(ctx);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      return signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createXmcFoldQuantizedBinary() {
  return std::make_unique<XmcFoldQuantizedBinary>();
}

} // namespace onnx_mlir
