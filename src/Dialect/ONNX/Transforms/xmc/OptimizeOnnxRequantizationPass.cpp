// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cmath>
#include <optional>
#include <utility>

using namespace mlir;

namespace {

/// Check if a type has a quantized element type
bool isQuantizedType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return false;
  return isa<quant::QuantizedType>(tensorType.getElementType());
}

/// Get constant scale (float) and zero_point (int64) from DQ/Q operands.
/// Returns both when each Value is a constant; otherwise nullopt. For Concat.
static std::optional<std::pair<float, int64_t>> getConstScaleAndZp(
    Value scaleVal, Value zpVal) {
  if (!scaleVal || !zpVal || isa<NoneType>(scaleVal.getType()) ||
      isa<NoneType>(zpVal.getType()))
    return std::nullopt;
  auto scaleCst = scaleVal.getDefiningOp<ONNXConstantOp>();
  auto zpCst = zpVal.getDefiningOp<ONNXConstantOp>();
  if (!scaleCst || !zpCst)
    return std::nullopt;
  auto scaleAttr = dyn_cast_or_null<ElementsAttr>(scaleCst.getValueAttr());
  auto zpAttr = dyn_cast_or_null<ElementsAttr>(zpCst.getValueAttr());
  if (!scaleAttr || !zpAttr)
    return std::nullopt;
  auto getFloat = [](ElementsAttr attr) -> std::optional<float> {
    if (attr.isSplat()) {
      if (isa<FloatType>(attr.getElementType()))
        return static_cast<float>(
            attr.getSplatValue<APFloat>().convertToDouble());
      return std::nullopt;
    }
    auto shapedTy = dyn_cast<ShapedType>(attr.getType());
    if (!shapedTy || !shapedTy.hasStaticShape() ||
        shapedTy.getNumElements() != 1)
      return std::nullopt;
    auto a = *attr.getValues<Attribute>().begin();
    if (auto f = dyn_cast<FloatAttr>(a))
      return static_cast<float>(f.getValueAsDouble());
    return std::nullopt;
  };
  auto getInt = [](ElementsAttr attr) -> std::optional<int64_t> {
    if (attr.isSplat()) {
      Type et = attr.getElementType();
      if (isa<FloatType>(et))
        return static_cast<int64_t>(
            std::llround(attr.getSplatValue<APFloat>().convertToDouble()));
      if (auto it = dyn_cast<IntegerType>(et)) {
        APInt api = attr.getSplatValue<APInt>();
        return it.isUnsigned() ? static_cast<int64_t>(api.getZExtValue())
                               : static_cast<int64_t>(api.getSExtValue());
      }
      return std::nullopt;
    }
    auto shapedTy = dyn_cast<ShapedType>(attr.getType());
    if (!shapedTy || !shapedTy.hasStaticShape() ||
        shapedTy.getNumElements() != 1)
      return std::nullopt;
    auto a = *attr.getValues<Attribute>().begin();
    if (auto f = dyn_cast<FloatAttr>(a))
      return static_cast<int64_t>(std::llround(f.getValueAsDouble()));
    if (auto i = dyn_cast<IntegerAttr>(a))
      return static_cast<int64_t>(i.getInt());
    return std::nullopt;
  };
  std::optional<float> s = getFloat(scaleAttr);
  std::optional<int64_t> z = getInt(zpAttr);
  if (s && z)
    return std::make_pair(*s, *z);
  return std::nullopt;
}

/// Pattern for Q(parent) -> DQ(parent) -> op -> Q(output) patterns
/// Optimizes by updating only DQ(parent) to use Q(output)'s scale/zp.
/// Q(parent) is left unchanged, preserving the original quantization.
/// The DQ now dequantizes using the output's parameters, which means
/// the data-movement op and Q(output) will use matching parameters.
///
/// Before: Q(s1,zp1) -> DQ(s1,zp1) -> Reshape -> Q(s2,zp2)
/// After:  Q(s1,zp1) -> DQ(s2,zp2) -> Reshape -> Q(s2,zp2)
///
/// The DQ(s2,zp2) effectively performs the requantization: it interprets
/// Q(parent)'s integer output using the new scale/zp, producing f32 values
/// that when re-quantized by Q(output) with the same s2/zp2 yield the
/// correct result.
template <typename OpTy>
struct OnnxQDQRequantizationOptimizationPattern
    : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      OpTy op, PatternRewriter &rewriter) const override {
    // Get input value from op (should be f32, not quantized)
    ::mlir::Value inputValue = op->getOperand(0);

    // Verify op input is f32 (not quantized)
    auto inputType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(inputValue.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "Op input not ranked tensor");
    if (isQuantizedType(inputType))
      return rewriter.notifyMatchFailure(
          op, "Op input should be f32, not quantized");

    // Trace backwards to find DQ operation
    auto dqOpTyped = inputValue.getDefiningOp<::mlir::ONNXDequantizeLinearOp>();
    if (!dqOpTyped)
      return rewriter.notifyMatchFailure(op, "Input not from DequantizeLinear");
    if (!inputValue.hasOneUse())
      return rewriter.notifyMatchFailure(op, "DQ has multiple uses");

    // Get DQ's quantized input
    ::mlir::Value dqInput = dqOpTyped->getOperand(0);

    // Trace forwards to find output Q operation
    ::mlir::Value opResult = op.getResult();
    auto outputType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(opResult.getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(op, "Op output not ranked tensor");
    if (isQuantizedType(outputType))
      return rewriter.notifyMatchFailure(
          op, "Op output should be f32, not quantized");
    if (!opResult.hasOneUse())
      return rewriter.notifyMatchFailure(op, "Op result has multiple uses");

    auto qOutputOpTyped = ::mlir::dyn_cast<::mlir::ONNXQuantizeLinearOp>(
        *opResult.getUsers().begin());
    if (!qOutputOpTyped)
      return rewriter.notifyMatchFailure(
          op, "Op output not consumed by QuantizeLinear");

    // Get DQ(parent)'s and Q(output)'s scale/zp operands
    ::mlir::Value dqScaleVal = dqOpTyped->getOperand(1);
    ::mlir::Value dqZpVal = dqOpTyped->getOperand(2);
    ::mlir::Value qOutputScaleVal = qOutputOpTyped->getOperand(1);
    ::mlir::Value qOutputZpVal = qOutputOpTyped->getOperand(2);

    // Check if requantization is happening
    if (dqScaleVal == qOutputScaleVal && dqZpVal == qOutputZpVal)
      return rewriter.notifyMatchFailure(op, "No requantization detected");

    // Only modify DQ: recreate it with Q(output)'s scale/zp
    // Q(parent) stays unchanged
    auto dqOutputType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(inputValue.getType());
    if (!dqOutputType)
      return rewriter.notifyMatchFailure(op, "DQ output not ranked tensor");

    auto newDQOp = rewriter.create<::mlir::ONNXDequantizeLinearOp>(
        dqOpTyped->getLoc(), ::mlir::SmallVector<::mlir::Type>{dqOutputType},
        ::mlir::SmallVector<::mlir::Value>{
            dqInput, qOutputScaleVal, qOutputZpVal},
        dqOpTyped->getAttrs());

    // Recreate the data-movement op with new DQ output
    ::mlir::SmallVector<::mlir::Value> newOpOperands(op->getOperands());
    newOpOperands[0] = newDQOp.getResult();
    auto newOp = rewriter.create<OpTy>(op->getLoc(),
        ::mlir::SmallVector<::mlir::Type>{outputType}, newOpOperands,
        op->getAttrs());

    // Recreate Q(output) with the new op result (same scale/zp)
    auto qOutputResultType = ::mlir::dyn_cast<::mlir::RankedTensorType>(
        qOutputOpTyped->getResult(0).getType());
    if (!qOutputResultType)
      return rewriter.notifyMatchFailure(
          op, "Q(output) result not ranked tensor");

    auto newQOutputOp =
        rewriter.create<::mlir::ONNXQuantizeLinearOp>(qOutputOpTyped->getLoc(),
            ::mlir::SmallVector<::mlir::Type>{qOutputResultType},
            ::mlir::SmallVector<::mlir::Value>{
                newOp.getResult(), qOutputScaleVal, qOutputZpVal},
            qOutputOpTyped->getAttrs());

    // Replace old operations (Q(parent) is NOT replaced)
    rewriter.replaceOp(qOutputOpTyped, newQOutputOp);
    rewriter.replaceOp(op, newOp);
    rewriter.replaceOp(dqOpTyped, newDQOp);

    return ::mlir::success();
  }
};
/// Specialization for Concat operation (multiple inputs)
template <>
::mlir::LogicalResult
OnnxQDQRequantizationOptimizationPattern<::mlir::ONNXConcatOp>::matchAndRewrite(
    ::mlir::ONNXConcatOp op, ::mlir::PatternRewriter &rewriter) const {
  // Trace forwards to find output Q operation
  ::mlir::Value opResult = op.getResult();

  // Verify op output is f32 (not quantized)
  auto outputType =
      ::mlir::dyn_cast<::mlir::RankedTensorType>(opResult.getType());
  if (!outputType)
    return rewriter.notifyMatchFailure(op, "Op output not ranked tensor");

  // Check that op output is NOT quantized (should be f32)
  if (isQuantizedType(outputType))
    return rewriter.notifyMatchFailure(
        op, "Op output should be f32, not quantized");

  if (!opResult.hasOneUse())
    return rewriter.notifyMatchFailure(op, "Op result has multiple uses");

  ::mlir::Operation *qOutputOp = *opResult.getUsers().begin();
  auto qOutputOpTyped =
      ::mlir::dyn_cast<::mlir::ONNXQuantizeLinearOp>(qOutputOp);
  if (!qOutputOpTyped)
    return rewriter.notifyMatchFailure(
        op, "Op output not consumed by QuantizeLinear");

  // Get Q(output)'s operands for scale and zero_point
  ::mlir::Value qOutputScaleVal = qOutputOpTyped->getOperand(1);
  ::mlir::Value qOutputZpVal = qOutputOpTyped->getOperand(2);

  // Process each input: only modify DQ, leave Q(parent) unchanged
  bool anyUpdated = false;
  ::mlir::SmallVector<::mlir::Value> newInputs;

  for (auto inputValue : op.getInputs()) {
    // Verify input is f32 (not quantized)
    auto inputType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(inputValue.getType());
    if (!inputType || isQuantizedType(inputType)) {
      newInputs.push_back(inputValue);
      continue;
    }

    // Trace backwards to find DQ operation
    auto dqOpTyped = inputValue.getDefiningOp<::mlir::ONNXDequantizeLinearOp>();
    if (!dqOpTyped || !inputValue.hasOneUse()) {
      newInputs.push_back(inputValue);
      continue;
    }

    // Get DQ's scale/zp and compare with Q(output)'s. Like xcompiler pass:
    // if |in_scale - out_scale| < 1e-6 and same zp, ignore (no requantize).
    ::mlir::Value dqScaleVal = dqOpTyped->getOperand(1);
    ::mlir::Value dqZpVal = dqOpTyped->getOperand(2);

    bool need_requantize = true;
    auto in = getConstScaleAndZp(dqScaleVal, dqZpVal);
    auto out = getConstScaleAndZp(qOutputScaleVal, qOutputZpVal);
    if (in && out) {
      float scale_abs_diff = std::fabs(in->first - out->first);
      if (scale_abs_diff < 1e-6f && in->second == out->second)
        need_requantize = false;
    } else {
      need_requantize =
          (dqScaleVal != qOutputScaleVal || dqZpVal != qOutputZpVal);
    }
    if (!need_requantize) {
      newInputs.push_back(inputValue);
      continue;
    }

    // Get DQ's quantized input (from Q(parent) — left unchanged)
    ::mlir::Value dqInput = dqOpTyped->getOperand(0);

    // DQ output type remains f32
    auto dqOutputType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(inputValue.getType());
    if (!dqOutputType) {
      newInputs.push_back(inputValue);
      continue;
    }

    // Recreate only DQ with Q(output)'s scale/zp; Q(parent) stays unchanged
    auto newDQOp = rewriter.create<::mlir::ONNXDequantizeLinearOp>(
        dqOpTyped->getLoc(), ::mlir::SmallVector<::mlir::Type>{dqOutputType},
        ::mlir::SmallVector<::mlir::Value>{
            dqInput, qOutputScaleVal, qOutputZpVal},
        dqOpTyped->getAttrs());

    newInputs.push_back(newDQOp.getResult());
    rewriter.replaceOp(dqOpTyped, newDQOp);
    anyUpdated = true;
  }

  if (!anyUpdated)
    return rewriter.notifyMatchFailure(op, "No Q/DQ pairs could be updated");

  // Recreate Concat with updated inputs
  ::mlir::SmallVector<::mlir::Type> newResultTypes = {outputType};

  auto newOp = rewriter.create<::mlir::ONNXConcatOp>(
      op->getLoc(), newResultTypes, newInputs, op->getAttrs());

  // Update Q(output) to use the new Concat result
  // Keep Q(output)'s original output type (regular tensor, not quantized)
  auto qOutputResultType = ::mlir::dyn_cast<::mlir::RankedTensorType>(
      qOutputOpTyped->getResult(0).getType());
  if (!qOutputResultType)
    return rewriter.notifyMatchFailure(
        op, "Q(output) result not ranked tensor");

  // Check that Q(output) result is NOT quantized (should be regular tensor
  // like ui8)
  if (isQuantizedType(qOutputResultType))
    return rewriter.notifyMatchFailure(
        op, "Q(output) result should be regular tensor, not quantized");

  ::mlir::SmallVector<::mlir::Type> newQOutputResultTypes = {qOutputResultType};
  ::mlir::SmallVector<::mlir::Value> newQOutputOperands = {
      newOp.getResult(), qOutputScaleVal, qOutputZpVal};

  auto newQOutputOp = rewriter.create<::mlir::ONNXQuantizeLinearOp>(
      qOutputOpTyped->getLoc(), newQOutputResultTypes, newQOutputOperands,
      qOutputOpTyped->getAttrs());

  // Replace old operations
  rewriter.replaceOp(qOutputOpTyped, newQOutputOp);
  rewriter.replaceOp(op, newOp);

  return ::mlir::success();
}
} // namespace

namespace onnx_mlir {

struct OptimizeOnnxRequantizationPass
    : public PassWrapper<OptimizeOnnxRequantizationPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "optimize-onnx-requantization";
  }
  StringRef getDescription() const override {
    return "Optimize requantization in ONNX operations that don't change "
           "quantization semantics (Reshape, Transpose, Slice, Concat)";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns
        .add<OnnxQDQRequantizationOptimizationPattern<::mlir::ONNXReshapeOp>>(
            patterns.getContext());
    patterns
        .add<OnnxQDQRequantizationOptimizationPattern<::mlir::ONNXTransposeOp>>(
            patterns.getContext());
    patterns.add<OnnxQDQRequantizationOptimizationPattern<::mlir::ONNXSliceOp>>(
        patterns.getContext());
    patterns.add<
        OnnxQDQRequantizationOptimizationPattern<::mlir::ONNXDepthToSpaceOp>>(
        patterns.getContext());
    patterns.add<
        OnnxQDQRequantizationOptimizationPattern<::mlir::ONNXSpaceToDepthOp>>(
        patterns.getContext());
    patterns
        .add<OnnxQDQRequantizationOptimizationPattern<::mlir::ONNXConcatOp>>(
            patterns.getContext());
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createOptimizeOnnxRequantizationPass() {
  return std::make_unique<OptimizeOnnxRequantizationPass>();
}

} // namespace onnx_mlir