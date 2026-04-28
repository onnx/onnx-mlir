// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <cmath>
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
  if (!constType || constType.getNumElements() != 1)
    return {};

  auto constOp = rhs.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return {};

  auto attr = dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
  if (!attr)
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

bool isZPOutOfBounds(int64_t newZP, quant::UniformQuantizedType qType) {
  return ((newZP < qType.getStorageTypeMin()) ||
          (newZP > qType.getStorageTypeMax()));
}

template <typename ONNXBinOp>
quant::UniformQuantizedType getInQuantType(
    quant::UniformQuantizedType outQType, double binConst, Location loc) {
  double newScale = outQType.getScale();
  int64_t newZP = outQType.getZeroPoint();

  if constexpr (std::is_same_v<ONNXBinOp, ONNXAddOp>) {
    newZP += std::lround(binConst / newScale);
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXSubOp>) {
    newZP -= std::lround(binConst / newScale);
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXMulOp>) {
    if (binConst == 0.0)
      return nullptr;
    newScale /= binConst;
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXDivOp>) {
    newScale *= binConst;
  } else {
    static_assert(false, "Unsupported binary operation");
    return nullptr;
  }

  if (isZPOutOfBounds(newZP, outQType))
    return nullptr;

  return quant::UniformQuantizedType::getChecked(
      [&loc]() { return emitWarning(loc); }, outQType.getFlags(),
      outQType.getStorageType(), outQType.getExpressedType(), newScale, newZP,
      outQType.getStorageTypeMin(), outQType.getStorageTypeMax());
}

template <typename ONNXBinOp>
quant::UniformQuantizedType getOutQuantType(
    quant::UniformQuantizedType inQType, double binConst, Location loc) {
  double newScale = inQType.getScale();
  int64_t newZP = inQType.getZeroPoint();

  if constexpr (std::is_same_v<ONNXBinOp, ONNXAddOp>) {
    newZP -= std::lround(binConst / newScale);
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXSubOp>) {
    newZP += std::lround(binConst / newScale);
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXMulOp>) {
    newScale *= binConst;
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXDivOp>) {
    if (binConst == 0.0)
      return nullptr;
    newScale /= binConst;
  } else {
    static_assert(false, "Unsupported binary operation");
    return nullptr;
  }

  if (isZPOutOfBounds(newZP, inQType))
    return nullptr;

  return quant::UniformQuantizedType::getChecked(
      [&loc]() { return emitWarning(loc); }, inQType.getFlags(),
      inQType.getStorageType(), inQType.getExpressedType(), newScale, newZP,
      inQType.getStorageTypeMin(), inQType.getStorageTypeMax());
}

} // namespace

namespace onnx_mlir {

template <typename ONNXBinOp>
class FoldQuantized : public OpRewritePattern<ONNXBinOp> {
public:
  using OpRewritePattern<ONNXBinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXBinOp binOp, PatternRewriter &rewriter) const override {
    auto lhs = binOp->getOperand(0);
    auto rhs = binOp->getOperand(1);
    auto out = binOp->getResult(0);

    // No need of swapping inputs if first input is constant
    if (lhs.template getDefiningOp<ONNXConstantOp>())
      return rewriter.notifyMatchFailure(binOp, "LHS should not be a constant");

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto outType = dyn_cast<RankedTensorType>(out.getType());
    if (!lhsType || !outType)
      return rewriter.notifyMatchFailure(binOp, "Not Ranked TensorTypes");

    auto lhsQType =
        dyn_cast<quant::UniformQuantizedType>(lhsType.getElementType());
    auto outQType =
        dyn_cast<quant::UniformQuantizedType>(outType.getElementType());
    if (!lhsQType || !outQType)
      return rewriter.notifyMatchFailure(
          binOp, "Not Quantized input and output types");

    // No need of checking if constant passes through reshape, etc
    // Canonicalizations and Constant Folding takes care of those
    double binConst;
    if (auto constOpt = getConstant(rhs))
      binConst = *constOpt;
    else
      return rewriter.notifyMatchFailure(binOp, "RHS not scalar constant");

    // Either input/output with only ONNX ops are considered
    // scast or fused kernels will not be considered
    auto isONNXOp = [](Operation *op) -> bool {
      if (isa<ONNXDequantizeLinearOp, ONNXQuantizeLinearOp>(op))
        return false;
      return isa<ONNXDialect>(op->getDialect());
    };

    bool updateInput = isONNXOp(lhs.getDefiningOp()) && lhs.hasOneUse();
    bool updateOutput = llvm::all_of(out.getUsers(), isONNXOp);
    if (!updateInput && !updateOutput)
      return rewriter.notifyMatchFailure(
          binOp, "Cannot update quant params on either input or output");

    Location binLoc = binOp->getLoc();
    auto newQType =
        updateInput ? getInQuantType<ONNXBinOp>(outQType, binConst, binLoc)
                    : getOutQuantType<ONNXBinOp>(outQType, binConst, binLoc);
    if (!newQType)
      return rewriter.notifyMatchFailure(binOp, "Cannot get new QType");

    if (updateInput) {
      rewriter.modifyOpInPlace(
          binOp, [&]() { lhs.setType(lhsType.clone(newQType)); });
      auto scast = rewriter.create<quant::StorageCastOp>(
          binLoc, lhsType.clone(newQType.getStorageType()), lhs);
      auto replOp =
          rewriter.create<quant::StorageCastOp>(binLoc, outType, scast);
      rewriter.replaceOp(binOp, replOp);
    } else {
      auto scast = rewriter.create<quant::StorageCastOp>(
          binLoc, lhsType.clone(lhsQType.getStorageType()), lhs);
      auto replOp = rewriter.create<quant::StorageCastOp>(
          binLoc, outType.clone(newQType), scast);
      rewriter.replaceOp(binOp, replOp);
    }

    return success();
  }
};

class FoldQuantizedBinary
    : public PassWrapper<FoldQuantizedBinary, OperationPass<func::FuncOp>> {
public:
  [[nodiscard]] StringRef getArgument() const override {
    return "fold-quantized-binary";
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FoldQuantized<ONNXAddOp>, FoldQuantized<ONNXSubOp>,
        FoldQuantized<ONNXMulOp>, FoldQuantized<ONNXDivOp>>(ctx);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      return signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createFoldQuantizedBinary() {
  return std::make_unique<FoldQuantizedBinary>();
}

} // namespace onnx_mlir
