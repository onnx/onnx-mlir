// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/// Structure to hold quantization parameters for both per-tensor and per-channel
struct QuantParams {
  bool isPerChannel = false;
  // Per-tensor params
  double scale = 0.0;
  int64_t zeroPoint = 0;
  // Per-channel params
  SmallVector<double> scales;
  SmallVector<int64_t> zeroPoints;
  int32_t quantizedDimension = 0;

  bool operator==(const QuantParams &other) const {
    if (isPerChannel != other.isPerChannel)
      return false;
    if (isPerChannel) {
      if (quantizedDimension != other.quantizedDimension)
        return false;
      if (scales.size() != other.scales.size())
        return false;
      for (size_t i = 0; i < scales.size(); ++i) {
        if (std::abs(scales[i] - other.scales[i]) > 1e-6)
          return false;
        if (zeroPoints[i] != other.zeroPoints[i])
          return false;
      }
      return true;
    }
    return std::abs(scale - other.scale) < 1e-6 && zeroPoint == other.zeroPoint;
  }

  bool operator!=(const QuantParams &other) const { return !(*this == other); }
};

/// Extract quantization parameters from a quant.uniform type (per-tensor or
/// per-channel)
std::optional<QuantParams> getQuantParams(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return std::nullopt;

  Type elementType = tensorType.getElementType();

  // Try per-tensor quantization first
  if (auto quantType = dyn_cast<quant::UniformQuantizedType>(elementType)) {
    QuantParams params;
    params.isPerChannel = false;
    params.scale = quantType.getScale();
    params.zeroPoint = quantType.getZeroPoint();
    return params;
  }

  // Try per-channel quantization
  if (auto quantType =
          dyn_cast<quant::UniformQuantizedPerAxisType>(elementType)) {
    QuantParams params;
    params.isPerChannel = true;
    params.quantizedDimension = quantType.getQuantizedDimension();
    params.scales.assign(quantType.getScales().begin(),
        quantType.getScales().end());
    params.zeroPoints.assign(quantType.getZeroPoints().begin(),
        quantType.getZeroPoints().end());
    return params;
  }

  return std::nullopt;
}

/// Create a new tensor type with updated quantization parameters (per-tensor)
RankedTensorType updateQuantParamsPerTensor(RankedTensorType tensorType,
    double newScale, int64_t newZeroPoint, MLIRContext * /*ctx*/) {
  auto oldQuantType =
      dyn_cast<quant::UniformQuantizedType>(tensorType.getElementType());
  if (!oldQuantType)
    return tensorType;

  auto newQuantType = quant::UniformQuantizedType::get(oldQuantType.getFlags(),
      oldQuantType.getStorageType(), oldQuantType.getExpressedType(), newScale,
      newZeroPoint, oldQuantType.getStorageTypeMin(),
      oldQuantType.getStorageTypeMax());

  return RankedTensorType::get(tensorType.getShape(), newQuantType);
}

/// Create a new tensor type with updated quantization parameters (per-channel)
RankedTensorType updateQuantParamsPerChannel(RankedTensorType tensorType,
    ArrayRef<double> newScales, ArrayRef<int64_t> newZeroPoints,
    int32_t quantizedDimension, MLIRContext * /*ctx*/) {
  auto oldQuantType =
      dyn_cast<quant::UniformQuantizedPerAxisType>(tensorType.getElementType());
  if (!oldQuantType)
    return tensorType;

  auto newQuantType = quant::UniformQuantizedPerAxisType::get(
      oldQuantType.getFlags(), oldQuantType.getStorageType(),
      oldQuantType.getExpressedType(), newScales, newZeroPoints,
      quantizedDimension, oldQuantType.getStorageTypeMin(),
      oldQuantType.getStorageTypeMax());

  return RankedTensorType::get(tensorType.getShape(), newQuantType);
}

/// Create a new tensor type with updated quantization parameters (unified)
RankedTensorType updateQuantParams(RankedTensorType tensorType,
    const QuantParams &newParams, MLIRContext *ctx) {
  if (newParams.isPerChannel) {
    return updateQuantParamsPerChannel(tensorType, newParams.scales,
        newParams.zeroPoints, newParams.quantizedDimension, ctx);
  }
  return updateQuantParamsPerTensor(
      tensorType, newParams.scale, newParams.zeroPoint, ctx);
}

/// Pattern for ONNX operations that don't change quantization semantics
/// (e.g., Reshape, Transpose, Slice, DepthToSpace, SpaceToDepth)
template <typename OpTy>
struct OnnxRequantizationOptimizationPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      OpTy op, PatternRewriter &rewriter) const override {
    // Get input value - use getOperand(0) which works for all single-input ops

    ::mlir::Value inputValue = op->getOperand(0);

    // Get input and output types
    auto inputType = dyn_cast<RankedTensorType>(inputValue.getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "Not ranked tensor types");

    // Extract quantization parameters (supports both per-tensor and per-channel)
    auto inputQuant = getQuantParams(inputType);
    auto outputQuant = getQuantParams(outputType);

    if (!inputQuant || !outputQuant)
      return rewriter.notifyMatchFailure(op, "Not quantized types");

    // Check if quantization types are compatible (both per-tensor or both
    // per-channel)
    if (inputQuant->isPerChannel != outputQuant->isPerChannel)
      return rewriter.notifyMatchFailure(
          op, "Incompatible quantization types (per-tensor vs per-channel)");

    // Check if requantization is happening
    if (*inputQuant == *outputQuant)
      return rewriter.notifyMatchFailure(op, "No requantization detected");

    // Get parent operation
    Operation *parentOp = inputValue.getDefiningOp();

    if (!parentOp)
      return rewriter.notifyMatchFailure(op, "No parent operation");

    // Check single use constraint
    if (!inputValue.hasOneUse())
      return rewriter.notifyMatchFailure(op, "Parent has multiple uses");

    // Update parent's output type to match current op's output quantization
    // Note: inputType is already verified to be RankedTensorType above
    auto newParentResultType =
        updateQuantParams(inputType, *outputQuant, rewriter.getContext());
    inputValue.setType(newParentResultType);

    // Recreate the operation with updated input type
    SmallVector<Type> newResultTypes = {outputType};
    SmallVector<Value> newOperands = op->getOperands();

    auto newOp = rewriter.create<OpTy>(
        op->getLoc(), newResultTypes, newOperands, op->getAttrs());

    rewriter.replaceOp(op, newOp);

    return success();
  }
};

/// Specialization for Concat operation (multiple inputs)
template <>
LogicalResult
OnnxRequantizationOptimizationPattern<ONNXConcatOp>::matchAndRewrite(
    ONNXConcatOp op, PatternRewriter &rewriter) const {
  // Get output type
  auto outputType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!outputType)
    return rewriter.notifyMatchFailure(op, "Output not ranked tensor type");

  // Extract output quantization parameters (supports both per-tensor and
  // per-channel)
  auto outputQuant = getQuantParams(outputType);
  if (!outputQuant)
    return rewriter.notifyMatchFailure(op, "Output not quantized");

  // Check all inputs for requantization opportunities
  bool hasRequantization = false;
  SmallVector<Value> inputValues;

  for (auto input : op.getInputs()) {
    inputValues.push_back(input);

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      continue;

    auto inputQuant = getQuantParams(inputType);
    if (!inputQuant)
      continue;

    // Skip incompatible quantization types
    if (inputQuant->isPerChannel != outputQuant->isPerChannel)
      continue;

    // Check if requantization is happening
    if (*inputQuant != *outputQuant) {
      hasRequantization = true;
      break;
    }
  }

  if (!hasRequantization)
    return rewriter.notifyMatchFailure(op, "No requantization detected");

  // Process each input and update parent operations
  bool anyUpdated = false;
  for (auto inputValue : inputValues) {
    auto inputType = dyn_cast<RankedTensorType>(inputValue.getType());
    if (!inputType)
      continue;

    auto inputQuant = getQuantParams(inputType);
    if (!inputQuant)
      continue;

    // Skip incompatible quantization types
    if (inputQuant->isPerChannel != outputQuant->isPerChannel)
      continue;

    // Skip if already matches output quantization
    if (*inputQuant == *outputQuant)
      continue;

    // Get parent operation
    Operation *parentOp = inputValue.getDefiningOp();
    if (!parentOp)
      continue;

    // Check single use constraint
    if (!inputValue.hasOneUse())
      continue;

    auto parentResultType = dyn_cast<RankedTensorType>(inputValue.getType());
    if (!parentResultType)
      continue;

    // Update parent's output type to match Concat's output quantization
    auto newParentResultType =
        updateQuantParams(parentResultType, *outputQuant, rewriter.getContext());
    inputValue.setType(newParentResultType);
    anyUpdated = true;
  }

  if (!anyUpdated)
    return rewriter.notifyMatchFailure(
        op, "No parent operations could be updated");

  // Recreate Concat with updated input types
  SmallVector<Type> newResultTypes = {outputType};
  SmallVector<Value> newOperands(op.getInputs().begin(), op.getInputs().end());

  auto newOp = rewriter.create<ONNXConcatOp>(
      op->getLoc(), newResultTypes, newOperands, op->getAttrs());

  rewriter.replaceOp(op, newOp);

  return success();
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

    // Register patterns for Reshape, Transpose, Slice, and Concat
    patterns.add<OnnxRequantizationOptimizationPattern<ONNXReshapeOp>>(context);
    patterns.add<OnnxRequantizationOptimizationPattern<ONNXTransposeOp>>(
        context);
    patterns.add<OnnxRequantizationOptimizationPattern<ONNXSliceOp>>(context);
    patterns.add<OnnxRequantizationOptimizationPattern<ONNXConcatOp>>(context);
    patterns.add<OnnxRequantizationOptimizationPattern<ONNXDepthToSpaceOp>>(context);
    patterns.add<OnnxRequantizationOptimizationPattern<ONNXSpaceToDepthOp>>(context);


    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createOptimizeOnnxRequantizationPass() {
  return std::make_unique<OptimizeOnnxRequantizationPass>();
}

} // namespace onnx_mlir
