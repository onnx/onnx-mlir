// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights reserved.

#include <memory>
#include <variant>

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "ResultNamesUpdater.hpp"
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
    }
    ();
  }

  bool isSigned =
      storageType.isSignedInteger() || storageType.isSignlessInteger();

  if (scale.getNumElements() == 1 && zeropoint.getNumElements() == 1) {
    return quant::UniformQuantizedType::get(isSigned, storageType,
        expressedType,
        scale.template getSplatValue<APFloat>().convertToDouble(),
        storageType.isSignedInteger()
            ? zeropoint.template getSplatValue<APInt>().getSExtValue()
            : zeropoint.template getSplatValue<APInt>().getZExtValue(),
        quant::QuantizedType::getDefaultMinimumForInteger(
            isSigned, storageType.getIntOrFloatBitWidth()),
        quant::QuantizedType::getDefaultMaximumForInteger(
            isSigned, storageType.getIntOrFloatBitWidth()));
  } else if (op.getBlockSize() == 0) {
    SmallVector<double> scales(scale.getNumElements());
    llvm::transform(scale.template getValues<APFloat>(), scales.begin(),
        [](APFloat apFloat) { return apFloat.convertToDouble(); });
    SmallVector<int64_t> zeropoints(zeropoint.getNumElements());
    llvm::transform(zeropoint.template getValues<APInt>(), zeropoints.begin(),
        [storageType](APInt apInt) {
          return storageType.isSignedInteger() ? apInt.getSExtValue()
                                               : apInt.getZExtValue();
        });
    return quant::UniformQuantizedPerAxisType::get(isSigned, storageType,
        expressedType, scales, zeropoints, op.getAxis(),
        quant::QuantizedType::getDefaultMinimumForInteger(
            isSigned, storageType.getIntOrFloatBitWidth()),
        quant::QuantizedType::getDefaultMaximumForInteger(
            isSigned, storageType.getIntOrFloatBitWidth()));
  }

  // TODO: Add support for blockwise quantization
  return StringLiteral("Blockwise quantization not supported");
}

} // namespace

class DQToSCast : public OpRewritePattern<ONNXDequantizeLinearOp> {
public:
  using OpRewritePattern<ONNXDequantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXDequantizeLinearOp dqOp, PatternRewriter &rewriter) const override {
    if (llvm::any_of(dqOp.getY().getUsers(), [](Operation *op) {
          return op->hasTrait<OpTrait::IsTerminator>();
        })) {
      return rewriter.notifyMatchFailure(
          dqOp, "Cannot convert DQ output to function return");
    }

    auto qTypeErr = getQuantType(dqOp);
    if (std::holds_alternative<StringLiteral>(qTypeErr))
      return rewriter.notifyMatchFailure(
          dqOp, std::get<StringLiteral>(qTypeErr));

    auto qType = std::get<quant::QuantizedType>(qTypeErr);
    auto qTensorType = cast<TensorType>(dqOp.getType()).clone(qType);
    if (auto constOp = dqOp.getX().getDefiningOp<ONNXConstantOp>();
        constOp && constOp.getResult().hasOneUse()) {
      rewriter.modifyOpInPlace(
          constOp, [&]() { constOp.getResult().setType(qTensorType); });
      rewriter.replaceOp(dqOp, constOp);
      return success();
    }

    auto scast = rewriter.create<quant::StorageCastOp>(
        dqOp.getLoc(), qTensorType, dqOp.getX());
    ResultNamesUpdater().notifyOperationReplaced(dqOp, scast.getResult());
    rewriter.replaceOp(dqOp, scast);
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

    // Copy the ResultName of qOp to parentOp
    ResultNamesUpdater().notifyOperationReplaced(qOp, qOp.getX());

    auto scast = rewriter.create<quant::StorageCastOp>(
        qOp.getLoc(), qOp.getY().getType(), qOp.getX());
    rewriter.replaceOp(qOp, scast);
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
