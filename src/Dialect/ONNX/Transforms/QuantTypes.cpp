// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights reserved.

#include <memory>

#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/STLExtras.h"

namespace onnx_mlir {

class Dequantize : public mlir::OpRewritePattern<mlir::ONNXDequantizeLinearOp> {
  using mlir::OpRewritePattern<mlir::ONNXDequantizeLinearOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXDequantizeLinearOp dqOp,
      mlir::PatternRewriter &rewriter) const override {

    // Should not convert types when the output is used by 'return' op
    if (llvm::any_of(dqOp.getY().getUsers(), [](mlir::Operation *op) {
          return mlir::isa<mlir::func::ReturnOp>(op);
        }))
      return rewriter.notifyMatchFailure(dqOp, "Not converting return op");

    if (!mlir::isa_and_present<mlir::ONNXConstantOp>(
            dqOp.getXScale().getDefiningOp()) ||
        !mlir::isa_and_present<mlir::ONNXConstantOp>(
            dqOp.getXZeroPoint().getDefiningOp()))
      return rewriter.notifyMatchFailure(dqOp, "Scale/Zeropoint not constant");

    auto scale = mlir::dyn_cast<mlir::DenseIntOrFPElementsAttr>(
        mlir::cast<mlir::ONNXConstantOp>(dqOp.getXScale().getDefiningOp())
            .getValueAttr());
    auto zeropoint = mlir::dyn_cast<mlir::DenseIntOrFPElementsAttr>(
        mlir::cast<mlir::ONNXConstantOp>(dqOp.getXZeroPoint().getDefiningOp())
            .getValueAttr());
    if (scale == nullptr || zeropoint == nullptr)
      return rewriter.notifyMatchFailure(
          dqOp, "Scale/Zeropoint not DenseElementsAttr");

    // TODO: Add support for per-channel quantization
    if (scale.getNumElements() != 1 || zeropoint.getNumElements() != 1)
      return rewriter.notifyMatchFailure(dqOp, "Not scalar scale or zeropoint");

    auto storageType =
        mlir::cast<mlir::TensorType>(dqOp.getX().getType()).getElementType();
    unsigned flags = storageType.isSignedInteger();
    auto resultType = mlir::cast<mlir::TensorType>(dqOp.getY().getType());
    auto expressedType = resultType.getElementType();

    // Check if already converted DQ, and fold previous nodes as needed
    if (mlir::isa<mlir::quant::QuantizedType>(expressedType)) {
      auto *inOp = dqOp.getX().getDefiningOp();
      // Rewrite Q -> DQ for which the types are already converted
      if (auto qOp =
              mlir::dyn_cast_if_present<mlir::ONNXQuantizeLinearOp>(inOp);
          qOp && qOp.getX().getType() == resultType) {
        rewriter.replaceOp(dqOp, qOp.getX());
        return mlir::success();
      }
      // Rewrite const -> DQ
      if (auto constOp =
              mlir::dyn_cast_if_present<mlir::ONNXConstantOp>(inOp)) {
        rewriter.replaceOpWithNewOp<mlir::ONNXConstantOp>(dqOp,
            mlir::TypeRange({resultType}), mlir::ValueRange(),
            mlir::SmallVector<mlir::NamedAttribute>{
                {"value", constOp.getValueAttr()}});
        return mlir::success();
      }
      return rewriter.notifyMatchFailure(dqOp, "Already converted");
    }

    mlir::Type quantType = mlir::quant::UniformQuantizedType::get(flags,
        storageType, expressedType,
        scale.getSplatValue<mlir::APFloat>().convertToDouble(),
        storageType.isSignedInteger()
            ? zeropoint.getSplatValue<mlir::APInt>().getSExtValue()
            : zeropoint.getSplatValue<mlir::APInt>().getZExtValue(),
        mlir::quant::QuantizedType::getDefaultMinimumForInteger(
            storageType.isSignedInteger(), storageType.getIntOrFloatBitWidth()),
        mlir::quant::QuantizedType::getDefaultMaximumForInteger(
            storageType.isSignedInteger(),
            storageType.getIntOrFloatBitWidth()));

    // Change the result type of this op
    rewriter.modifyOpInPlace(
        dqOp, [&]() { dqOp.getY().setType(resultType.clone(quantType)); });

    return mlir::success();
  }
};

class Quantize : public mlir::OpRewritePattern<mlir::ONNXQuantizeLinearOp> {
  using mlir::OpRewritePattern<mlir::ONNXQuantizeLinearOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXQuantizeLinearOp qOp,
      mlir::PatternRewriter &rewriter) const override {
    // Should not convert when input comes from blockArg or Cast node
    // TODO: Fix for cast node by either of:
    //  - Updating the cast to quantized type
    //  - Add qcast after cast
    auto *inOp = qOp.getX().getDefiningOp();
    if (inOp == nullptr || mlir::isa<mlir::ONNXCastOp>(inOp))
      return rewriter.notifyMatchFailure(
          qOp, "Not converting blockArg or Cast output");

    if (!mlir::isa_and_present<mlir::ONNXConstantOp>(
            qOp.getYScale().getDefiningOp()) ||
        !mlir::isa_and_present<mlir::ONNXConstantOp>(
            qOp.getYZeroPoint().getDefiningOp()))
      return rewriter.notifyMatchFailure(qOp, "Scale/Zeropoint not constant");

    auto scale = mlir::dyn_cast<mlir::DenseIntOrFPElementsAttr>(
        mlir::cast<mlir::ONNXConstantOp>(qOp.getYScale().getDefiningOp())
            .getValueAttr());
    auto zeropoint = mlir::dyn_cast<mlir::DenseIntOrFPElementsAttr>(
        mlir::cast<mlir::ONNXConstantOp>(qOp.getYZeroPoint().getDefiningOp())
            .getValueAttr());
    if (scale == nullptr || zeropoint == nullptr)
      return rewriter.notifyMatchFailure(
          qOp, "Scale/Zeropoint not DenseElementsAttr");

    // TODO: Add support for per-channel quantization
    if (scale.getNumElements() != 1 || zeropoint.getNumElements() != 1)
      return rewriter.notifyMatchFailure(qOp, "Not scalar scale or zeropoint");

    auto storageType =
        mlir::cast<mlir::TensorType>(qOp.getY().getType()).getElementType();
    unsigned flags = storageType.isSignedInteger();
    auto operandType = mlir::cast<mlir::TensorType>(qOp.getX().getType());
    auto expressedType = operandType.getElementType();

    if (mlir::isa<mlir::quant::QuantizedType>(expressedType))
      return rewriter.notifyMatchFailure(qOp, "Already converted");

    mlir::Type quantType = mlir::quant::UniformQuantizedType::get(flags,
        storageType, expressedType,
        scale.getSplatValue<mlir::APFloat>().convertToDouble(),
        storageType.isSignedInteger()
            ? zeropoint.getSplatValue<mlir::APInt>().getSExtValue()
            : zeropoint.getSplatValue<mlir::APInt>().getZExtValue(),
        mlir::quant::QuantizedType::getDefaultMinimumForInteger(
            storageType.isSignedInteger(), storageType.getIntOrFloatBitWidth()),
        mlir::quant::QuantizedType::getDefaultMaximumForInteger(
            storageType.isSignedInteger(),
            storageType.getIntOrFloatBitWidth()));

    // Change the result type of this op
    rewriter.modifyOpInPlace(
        qOp, [&]() { qOp.getX().setType(operandType.clone(quantType)); });

    return mlir::success();
  }
};

class QuantTypesPass : public mlir::PassWrapper<QuantTypesPass,
                           mlir::OperationPass<mlir::func::FuncOp>> {
  mlir::StringRef getArgument() const override { return "quant-types"; }

  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<Dequantize, Quantize>(ctx);
    if (failed(mlir::applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createQuantTypesPass() {
  return std::make_unique<QuantTypesPass>();
}

} // namespace onnx_mlir
