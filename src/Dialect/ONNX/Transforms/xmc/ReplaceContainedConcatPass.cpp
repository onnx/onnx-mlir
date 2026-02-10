// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

// Check if all values in subset are contained in superset.
static bool isSubset(ValueRange superset, ValueRange subset) {
  llvm::DenseSet<Value> supersetSet(superset.begin(), superset.end());
  for (Value v : subset)
    if (!supersetSet.contains(v))
      return false;
  return true;
}

// Check if quantization parameters match between inputs and output.
static bool isFixMatch(ONNXConcatOp concatOp) {
  auto outputType = dyn_cast<ShapedType>(concatOp.getResult().getType());
  if (!outputType)
    return false;

  auto outputQType =
      dyn_cast<quant::UniformQuantizedType>(outputType.getElementType());

  for (Value input : concatOp.getInputs()) {
    auto inputType = dyn_cast<ShapedType>(input.getType());
    if (!inputType)
      return false;

    auto inputQType =
        dyn_cast<quant::UniformQuantizedType>(inputType.getElementType());

    if (outputQType && inputQType) {
      if (outputQType.getScale() != inputQType.getScale() ||
          outputQType.getZeroPoint() != inputQType.getZeroPoint())
        return false;
    } else if (outputQType || inputQType) {
      return false;
    }
  }
  return true;
}

// Create new input list: [innerConcatOutput] + (outerInputs - innerInputs)
static SmallVector<Value> createNewConcatInputs(
    ValueRange outerInputs, ValueRange innerInputs, Value innerConcatOutput) {
  SmallVector<Value> newInputs;
  newInputs.push_back(innerConcatOutput);

  llvm::DenseSet<Value> innerSet(innerInputs.begin(), innerInputs.end());
  for (Value input : outerInputs)
    if (!innerSet.contains(input))
      newInputs.push_back(input);
  return newInputs;
}

// Implements the contained-concat optimization.
struct ReplaceContainedConcatPattern : public OpRewritePattern<ONNXConcatOp> {
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConcatOp innerConcatOp, PatternRewriter &rewriter) const override {
    if (!isFixMatch(innerConcatOp))
      return failure();

    auto innerInputs = innerConcatOp.getInputs();
    if (innerInputs.empty())
      return failure();

    // Normalize negative axis based on ranked type.
    auto getNormalizedAxis = [](ONNXConcatOp op) -> std::optional<int64_t> {
      auto axisAttr = op.getAxisAttr();
      if (!axisAttr)
        return std::nullopt;
      auto rankedTy = dyn_cast<RankedTensorType>(op.getType());
      if (!rankedTy)
        return std::nullopt;

      int64_t axis = axisAttr.getSInt();
      int64_t rank = rankedTy.getRank();
      if (rank <= 0)
        return std::nullopt;
      if (axis < 0)
        axis += rank;
      if (axis < 0 || axis >= rank)
        return std::nullopt;
      return axis;
    };

    auto innerAxisOpt = getNormalizedAxis(innerConcatOp);
    if (!innerAxisOpt)
      return failure();
    int64_t innerAxis = *innerAxisOpt;

    // Look at the users of the first input of innerConcat.
    Value firstInput = innerInputs[0];
    for (Operation *user : firstInput.getUsers()) {
      auto outerConcatOp = dyn_cast<ONNXConcatOp>(user);
      if (!outerConcatOp || outerConcatOp == innerConcatOp)
        continue;

      if (!isFixMatch(outerConcatOp))
        continue;

      auto outerAxisOpt = getNormalizedAxis(outerConcatOp);
      if (!outerAxisOpt || *outerAxisOpt != innerAxis)
        continue;

      auto outerInputs = outerConcatOp.getInputs();
      if (!isSubset(outerInputs, innerInputs))
        continue;

      SmallVector<Value> newInputs = createNewConcatInputs(
          outerInputs, innerInputs, innerConcatOp.getResult());
      if (newInputs.size() >= outerInputs.size())
        continue;

      rewriter.setInsertionPoint(outerConcatOp);
      auto si64Type =
          IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);
      auto axisAttr = IntegerAttr::get(si64Type, innerAxis);
      auto newConcatOp = rewriter.create<ONNXConcatOp>(
          outerConcatOp.getLoc(), outerConcatOp.getType(), newInputs, axisAttr);

      rewriter.replaceOp(outerConcatOp, newConcatOp.getResult());
      return success();
    }

    return failure();
  }
};

} // namespace

namespace onnx_mlir {

struct ReplaceContainedConcatPass
    : public PassWrapper<ReplaceContainedConcatPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "replace-contained-concat"; }
  StringRef getDescription() const override {
    return "Optimize concat operations by reusing subset concat results";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReplaceContainedConcatPattern>(context);

    GreedyRewriteConfig config;
    config.maxIterations = 10;
    config.useTopDownTraversal = false;

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createReplaceContainedConcatPass() {
  return std::make_unique<ReplaceContainedConcatPass>();
}

} // namespace onnx_mlir
