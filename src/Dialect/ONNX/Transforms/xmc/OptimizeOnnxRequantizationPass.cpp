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

/// Extract quantization parameters from a quant.uniform type
std::optional<std::pair<double, int64_t>> getQuantParams(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return std::nullopt;

  auto quantType =
      dyn_cast<quant::UniformQuantizedType>(tensorType.getElementType());
  if (!quantType)
    return std::nullopt;

  return std::make_pair(quantType.getScale(), quantType.getZeroPoint());
}

/// Create a new tensor type with updated quantization parameters
RankedTensorType updateQuantParams(RankedTensorType tensorType, double newScale,
    int64_t newZeroPoint, MLIRContext * /*ctx*/) {
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

/// Pattern for ONNX operations that don't change quantization semantics
/// (e.g., Reshape, Transpose, Slice)
template <typename OpTy>
struct OnnxRequantizationOptimizationPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      OpTy op, PatternRewriter &rewriter) const override {
    // Get input and output types
    auto inputType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "Not ranked tensor types");

    // Extract quantization parameters
    auto inputQuant = getQuantParams(inputType);
    auto outputQuant = getQuantParams(outputType);

    if (!inputQuant || !outputQuant)
      return rewriter.notifyMatchFailure(op, "Not quantized types");

    auto [inputScale, inputZp] = *inputQuant;
    auto [outputScale, outputZp] = *outputQuant;

    // Check if requantization is happening
    if (std::abs(inputScale - outputScale) < 1e-6 && inputZp == outputZp)
      return rewriter.notifyMatchFailure(op, "No requantization detected");

    // Get parent operation
    Value inputValue = op.getData();
    Operation *parentOp = inputValue.getDefiningOp();

    if (!parentOp)
      return rewriter.notifyMatchFailure(op, "No parent operation");

    // Check single use constraint
    if (!inputValue.hasOneUse())
      return rewriter.notifyMatchFailure(op, "Parent has multiple uses");

    auto parentResultType = dyn_cast<RankedTensorType>(inputValue.getType());
    if (!parentResultType)
      return rewriter.notifyMatchFailure(op, "Parent result not ranked tensor");

    // Update parent's output type to match current op's output quantization
    auto newParentResultType = updateQuantParams(
        parentResultType, outputScale, outputZp, rewriter.getContext());
    inputValue.setType(newParentResultType);

    // Update current op's input type to match output quantization
    // Note: newInputType is not used directly, but the type update propagates
    // through the graph
    (void)updateQuantParams(
        inputType, outputScale, outputZp, rewriter.getContext());

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

  // Extract output quantization parameters
  auto outputQuant = getQuantParams(outputType);
  if (!outputQuant)
    return rewriter.notifyMatchFailure(op, "Output not quantized");

  auto [outputScale, outputZp] = *outputQuant;

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

    auto [inputScale, inputZp] = *inputQuant;

    // Check if requantization is happening
    if (std::abs(inputScale - outputScale) > 1e-6 || inputZp != outputZp) {
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

    auto [inputScale, inputZp] = *inputQuant;

    // Skip if already matches output quantization
    if (std::abs(inputScale - outputScale) < 1e-6 && inputZp == outputZp)
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
    auto newParentResultType = updateQuantParams(
        parentResultType, outputScale, outputZp, rewriter.getContext());
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

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createOptimizeOnnxRequantizationPass() {
  return std::make_unique<OptimizeOnnxRequantizationPass>();
}

} // namespace onnx_mlir
