// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "remove-useless-qlinear-pool"

using namespace mlir;

namespace {

/// Helper function to check if an I64ArrayAttr contains all ones
static bool isAllOnes(ArrayAttr attr) {
  if (!attr)
    return false;
  for (auto val : attr) {
    auto intAttr = mlir::dyn_cast<IntegerAttr>(val);
    if (!intAttr || intAttr.getInt() != 1)
      return false;
  }
  return true;
}

/// Helper function to check if pool operation is a no-op
/// (kernel_shape = [1,1,...] and strides = [1,1,...])
static bool isPoolNoOp(ArrayAttr kernelShape, ArrayAttr strides) {
  // Check kernel_shape is all ones
  if (!isAllOnes(kernelShape))
    return false;

  // Check strides is all ones
  if (!isAllOnes(strides))
    return false;

  return true;
}

/// Helper function to check if input and output shapes are equal
static bool areShapesEqual(Type inputType, Type outputType) {
  auto inputShapedType = mlir::dyn_cast<ShapedType>(inputType);
  auto outputShapedType = mlir::dyn_cast<ShapedType>(outputType);

  if (!inputShapedType || !outputShapedType)
    return false;

  if (!inputShapedType.hasStaticShape() || !outputShapedType.hasStaticShape())
    return false;

  return inputShapedType.getShape() == outputShapedType.getShape();
}

/// Helper function to check if two quantized types have the same scale and
/// zero point
static bool haveMatchingQuantParams(Type inputType, Type outputType) {
  auto inputTensorType = mlir::dyn_cast<TensorType>(inputType);
  auto outputTensorType = mlir::dyn_cast<TensorType>(outputType);

  if (!inputTensorType || !outputTensorType)
    return true; // Non-tensor types, skip quant check

  auto inputElementType = inputTensorType.getElementType();
  auto outputElementType = outputTensorType.getElementType();

  auto inputQuantType =
      mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inputElementType);
  auto outputQuantType =
      mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outputElementType);

  // If neither is quantized, that's fine
  if (!inputQuantType && !outputQuantType)
    return true;

  // If only one is quantized, they don't match
  if (!inputQuantType || !outputQuantType)
    return false;

  // Both are quantized, check scale and zero point match
  return inputQuantType.getScale() == outputQuantType.getScale() &&
         inputQuantType.getZeroPoint() == outputQuantType.getZeroPoint();
}

/// Pattern to remove useless AveragePool operations
/// When kernel_shape and strides are all 1s, the pool is a no-op
struct RemoveUselessAveragePoolPattern
    : public OpRewritePattern<ONNXAveragePoolOp> {
  using OpRewritePattern<ONNXAveragePoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAveragePoolOp poolOp, PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "remove-useless-qlinear-pool: Trying to match "
                            << poolOp << "\n");

    auto kernelShape = poolOp.getKernelShapeAttr();
    auto strides = poolOp.getStridesAttr();

    // Check if this pool operation is a no-op
    if (!isPoolNoOp(kernelShape, strides))
      return rewriter.notifyMatchFailure(
          poolOp, "Pool operation is not a no-op (kernel/strides not all 1s)");

    Type inputType = poolOp.getX().getType();
    Type outputType = poolOp.getY().getType();

    // Input and output shapes should be equal
    if (!areShapesEqual(inputType, outputType))
      return rewriter.notifyMatchFailure(
          poolOp, "Input and output shapes are not equal");

    // Quant type tensors should have same scale/zp
    if (!haveMatchingQuantParams(inputType, outputType))
      return rewriter.notifyMatchFailure(
          poolOp, "Quantization parameters (scale/zp) do not match");

    // Replace pool output with its input
    rewriter.replaceOp(poolOp, poolOp.getX());

    return success();
  }
};

/// Pattern to remove useless MaxPoolSingleOut operations
/// When kernel_shape and strides are all 1s, the pool is a no-op
struct RemoveUselessMaxPoolPattern
    : public OpRewritePattern<ONNXMaxPoolSingleOutOp> {
  using OpRewritePattern<ONNXMaxPoolSingleOutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMaxPoolSingleOutOp poolOp, PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "remove-useless-qlinear-pool: Trying to match "
                            << poolOp << "\n");

    auto kernelShape = poolOp.getKernelShapeAttr();
    auto strides = poolOp.getStridesAttr();

    // Check if this pool operation is a no-op
    if (!isPoolNoOp(kernelShape, strides))
      return rewriter.notifyMatchFailure(
          poolOp, "Pool operation is not a no-op (kernel/strides not all 1s)");

    Type inputType = poolOp.getX().getType();
    Type outputType = poolOp.getO_Y().getType();

    // Input and output shapes should be equal
    if (!areShapesEqual(inputType, outputType))
      return rewriter.notifyMatchFailure(
          poolOp, "Input and output shapes are not equal");

    // Quant type tensors should have same scale/zp
    if (!haveMatchingQuantParams(inputType, outputType))
      return rewriter.notifyMatchFailure(
          poolOp, "Quantization parameters (scale/zp) do not match");

    // Replace pool output with its input
    rewriter.replaceOp(poolOp, poolOp.getX());

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct RemoveUselessQLinearPoolPass
    : public PassWrapper<RemoveUselessQLinearPoolPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "remove-useless-qlinear-pool";
  }
  StringRef getDescription() const override {
    return "Remove useless pool operations where kernel_shape and strides are "
           "all 1s (no-op pools)";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RemoveUselessAveragePoolPattern>(context);
    patterns.add<RemoveUselessMaxPoolPattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createRemoveUselessQLinearPoolPass() {
  return std::make_unique<RemoveUselessQLinearPoolPass>();
}

} // namespace onnx_mlir
