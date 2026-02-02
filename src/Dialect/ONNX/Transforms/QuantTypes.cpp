// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights reserved.

#include <memory>
#include <variant>

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

/* NOTE:
 * This conversion pass does not touch types where Q/DQ's are missing.
 * There is no insertion of "fake" quantization nodes.
 * This may create ops taking quantized operands and producing fp32 results
 * or vice-versa
 */

namespace {
template <typename QDQOp>
std::variant<quant::QuantizedType, StringLiteral> getQuantType(QDQOp op) {

  auto scaleOp = op->getOperand(1).template getDefiningOp<ONNXConstantOp>();
  auto zeropointOp = op->getOperand(2).template getDefiningOp<ONNXConstantOp>();
  if (!scaleOp || !zeropointOp)
    return StringLiteral("Scale/Zeropoint not constant");

  auto scale =
      dyn_cast_if_present<DenseIntOrFPElementsAttr>(scaleOp.getValueAttr());
  auto zeropoint =
      dyn_cast_if_present<DenseIntOrFPElementsAttr>(zeropointOp.getValueAttr());
  if (!scale || !zeropoint)
    return StringLiteral("Scale/Zeropoint not DenseElementsAttr");

  Value input = op->getOperand(0);
  Value result = op->getResult(0);

  Type storageType;
  Type expressedType;
  if constexpr (std::is_same_v<QDQOp, ONNXDequantizeLinearOp>) {
    storageType = cast<TensorType>(input.getType()).getElementType();
    expressedType = cast<TensorType>(result.getType()).getElementType();
  } else if constexpr (std::is_same_v<QDQOp, ONNXQuantizeLinearOp>) {
    storageType = cast<TensorType>(result.getType()).getElementType();
    expressedType = cast<TensorType>(input.getType()).getElementType();
  } else {
    // Cannot directly use static_assert(false) before c++23
    // Creating a templated lambda and invoking immediately
    []<bool flag = false>() {
      static_assert(flag, "Only defined for DequantizeLinear & QuantizeLinear");
    }();
  }

  if (scale.getNumElements() == 1 && zeropoint.getNumElements() == 1)
    return quant::UniformQuantizedType::get(storageType.isSignedInteger(),
        storageType, expressedType,
        scale.template getSplatValue<APFloat>().convertToDouble(),
        storageType.isSignedInteger()
            ? zeropoint.template getSplatValue<APInt>().getSExtValue()
            : zeropoint.template getSplatValue<APInt>().getZExtValue(),
        quant::QuantizedType::getDefaultMinimumForInteger(
            storageType.isSignedInteger(), storageType.getIntOrFloatBitWidth()),
        quant::QuantizedType::getDefaultMaximumForInteger(
            storageType.isSignedInteger(),
            storageType.getIntOrFloatBitWidth()));

  // TODO: Add support for per-channel quantization
  return StringLiteral("Scale/Zeropoint not scalar");
}

} // namespace

class DQToSCast : public OpRewritePattern<ONNXDequantizeLinearOp> {
public:
  using OpRewritePattern<ONNXDequantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXDequantizeLinearOp dqOp, PatternRewriter &rewriter) const override {
    if (llvm::any_of(dqOp.getY().getUsers(),
            [](Operation *op) { return isa<func::ReturnOp>(op); })) {
      return rewriter.notifyMatchFailure(
          dqOp, "Cannot convert DQ output to function return");
    }

    auto qTypeErr = getQuantType(dqOp);
    if (std::holds_alternative<StringLiteral>(qTypeErr))
      return rewriter.notifyMatchFailure(
          dqOp, std::get<StringLiteral>(qTypeErr));

    auto qType = std::get<quant::QuantizedType>(qTypeErr);
    auto qTensorType = cast<TensorType>(dqOp.getType()).clone(qType);
    rewriter.replaceOpWithNewOp<quant::StorageCastOp>(
        dqOp, qTensorType, dqOp.getX());

    return success();
  }
};

class QToSCast : public OpRewritePattern<ONNXQuantizeLinearOp> {
public:
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {
    if (isa<BlockArgument>(qOp.getOperand(0))) {
      return rewriter.notifyMatchFailure(
          qOp, "Cannot convert Q input from BlockArg");
    }

    auto qTypeErr = getQuantType(qOp);
    if (std::holds_alternative<StringLiteral>(qTypeErr))
      return rewriter.notifyMatchFailure(
          qOp, std::get<StringLiteral>(qTypeErr));

    auto qType = std::get<quant::QuantizedType>(qTypeErr);
    auto qTensorType = cast<TensorType>(qOp.getType()).clone(qType);
    rewriter.modifyOpInPlace(qOp, [&]() { qOp.getX().setType(qTensorType); });
    rewriter.replaceOpWithNewOp<quant::StorageCastOp>(
        qOp, qOp.getY().getType(), qOp.getX());

    return success();
  }
};

class QuantTypesPass
    : public PassWrapper<QuantTypesPass, OperationPass<func::FuncOp>> {
  [[nodiscard]] StringRef getArgument() const override { return "quant-types"; }

  void getDependentDialects(::DialectRegistry &registry) const override {
    registry.insert<quant::QuantDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DQToSCast, QToSCast>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createQuantTypesPass() {
  return std::make_unique<QuantTypesPass>();
}

} // namespace onnx_mlir
