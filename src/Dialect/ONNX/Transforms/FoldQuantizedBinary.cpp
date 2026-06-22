// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

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

bool isZPOutOfBounds(int64_t newZP, quant::UniformQuantizedType qType) {
  return ((newZP < qType.getStorageTypeMin()) ||
          (newZP > qType.getStorageTypeMax()));
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

template <typename ONNXBinOp>
quant::UniformQuantizedType getInQuantType(
    quant::UniformQuantizedType outQType, double binConst, Location loc) {
  double newScale = outQType.getScale();
  int64_t newZP = outQType.getZeroPoint();

  if constexpr (std::is_same_v<ONNXBinOp, ONNXAddOp>) {
    newZP += std::round(binConst / newScale);
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXSubOp>) {
    newZP -= std::round(binConst / newScale);
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXMulOp>) {
    if (binConst == 0.0)
      return nullptr;
    newScale /= binConst;
    newScale = convertToExpressedType(newScale, outQType);
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXDivOp>) {
    newScale *= binConst;
    newScale = convertToExpressedType(newScale, outQType);
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
    newZP -= std::round(binConst / newScale);
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXSubOp>) {
    newZP += std::round(binConst / newScale);
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXMulOp>) {
    newScale *= binConst;
    newScale = convertToExpressedType(newScale, inQType);
  } else if constexpr (std::is_same_v<ONNXBinOp, ONNXDivOp>) {
    if (binConst == 0.0)
      return nullptr;
    newScale /= binConst;
    newScale = convertToExpressedType(newScale, inQType);
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
    // Canonicalization takes care of that
    if (lhs.template getDefiningOp<ONNXConstantOp>())
      return rewriter.notifyMatchFailure(binOp, "LHS should not be a constant");

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto outType = dyn_cast<RankedTensorType>(out.getType());
    if (!lhsType || !outType)
      return rewriter.notifyMatchFailure(binOp, "Not Ranked TensorTypes");

    // quant.scast cannot change shape; folding a broadcasting binary would
    // drop the broadcast and create a mismatched-shape scast.
    if (lhsType.getShape() != outType.getShape())
      return rewriter.notifyMatchFailure(
          binOp, "Cannot fold quantized binary with broadcasting operand");

    auto lhsQType =
        dyn_cast<quant::UniformQuantizedType>(lhsType.getElementType());
    auto outQType =
        dyn_cast<quant::UniformQuantizedType>(outType.getElementType());
    if (!lhsQType || !outQType)
      return rewriter.notifyMatchFailure(
          binOp, "Not Quantized input and output types");

    // No need of checking if constant passes through reshape, etc
    // Constant folding takes care of that
    double binConst;
    if (auto constOpt = getConstant(rhs))
      binConst = *constOpt;
    else
      return rewriter.notifyMatchFailure(binOp, "RHS not scalar constant");

    // Either input/output with only ONNX ops are considered
    // scast or fused kernels will not be considered
    auto isONNXOp = [](Operation *op) -> bool {
      if (!op || isa<ONNXDequantizeLinearOp, ONNXQuantizeLinearOp>(op))
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
                    : getOutQuantType<ONNXBinOp>(lhsQType, binConst, binLoc);
    if (!newQType)
      return rewriter.notifyMatchFailure(binOp, "Cannot get new QType");

    if (updateInput) {
      // Update the input op to have the right quant type and ResultNames
      rewriter.modifyOpInPlace(
          binOp, [&]() { lhs.setType(lhsType.clone(newQType)); });
      ResultNamesUpdater().notifyOperationReplaced(binOp, lhs.getDefiningOp());

      auto qScast = rewriter.create<quant::StorageCastOp>(
          binLoc, lhsType.clone(newQType.getStorageType()), lhs);
      // If original Q Scast exists, just replace it with the new one
      if (auto oqScast = dyn_cast<quant::StorageCastOp>(*binOp->user_begin());
          binOp->hasOneUse() && oqScast) {
        rewriter.replaceOp(oqScast, qScast);
        return success();
      }

      auto dqScast =
          rewriter.create<quant::StorageCastOp>(binLoc, outType, qScast);
      rewriter.replaceOp(binOp, dqScast);
      return success();
    } else {
      // If original DQ Scast exists, just replace it with new one
      if (auto odqScast = lhs.template getDefiningOp<quant::StorageCastOp>()) {
        auto dqScast = rewriter.create<quant::StorageCastOp>(
            binLoc, outType.clone(newQType), odqScast->getOperand(0));
        rewriter.replaceOp(binOp, dqScast);
        return success();
      }

      auto qScast = rewriter.create<quant::StorageCastOp>(
          binLoc, lhsType.clone(lhsQType.getStorageType()), lhs);
      auto dqScast = rewriter.create<quant::StorageCastOp>(
          binLoc, outType.clone(newQType), qScast);
      rewriter.replaceOp(binOp, dqScast);
      return success();

      // Since we fold DQ -> Bin -> Q -> DQ into DQ, we should not be
      // propagating the ResultNames of Q
    }
  }
};

/// Folds binary ops Add, Sub, Mul, Div with Q-DQ by adjusting the scale and
/// zero-point parameters. It works with 2 patterns:
/// Q -> DQ -> Add -> Q   => Q(x, oscale, ozp + round(C/oscale))
/// DQ -> Add -> Q -> DQ => DQ(x, iscale, izp - round(C/iscale))
/// iscale, izp refer to quant params on the input of binary op
/// oscale, ozp refer to quant params on the output of binary op
///
/// Converting above patterns to quant types
/// not_scast --iqType-> Add --oqType->
/// => not_scast --newQType-> scast -> scast --oqType->
/// --iqType-> Add --oqType-> not_scast
/// => --iqType-> scast -> scast --newQType-> not_scast
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

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createFoldQuantizedBinary() {
  return std::make_unique<FoldQuantizedBinary>();
}

} // namespace onnx_mlir
