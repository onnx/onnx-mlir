// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "remove-continuous-transpose-with-reshape"

using namespace mlir;

namespace {
/// Merge consecutive order numbers into one.
/// Example: [0,3,1,2] => [0,2,1]
/// This identifies groups of consecutive indices in the permutation and
/// collapses them into a single logical dimension.
SmallVector<int64_t> computeNewOrder(ArrayRef<int64_t> order) {
  SmallVector<int64_t> mergedData;

  size_t i = 0;
  while (i < order.size()) {
    size_t j = i + 1;
    // Find the end of the consecutive sequence
    while (j < order.size() && order[j] == order[j - 1] + 1) {
      j++;
    }
    // Keep only the first element of the consecutive group
    mergedData.push_back(order[i]);
    i = j;
  }

  // Remap to normalized indices [0, 1, 2, ...]
  SmallVector<int64_t> result = mergedData;
  llvm::sort(result);

  for (int64_t &i : result) {
    auto *it = llvm::find(mergedData, i);
    i = std::distance(mergedData.begin(), it);
  }

  return result;
}

/// Merge continuous shape dimensions according to the order permutation.
/// Example: shape = [1,96,4,16,16], order = [0,3,4,1,2]
///          output_shape = [1, 96*4, 16*16] = [1, 384, 256]
SmallVector<int64_t> computeNewShape(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> order) {
  SmallVector<int64_t> result;

  size_t i = 0;
  while (i < order.size()) {
    size_t j = i + 1;
    int64_t data = shape[i];
    // Multiply dimensions that are part of a consecutive group
    while (j < order.size() && order[j] == order[j - 1] + 1) {
      data *= shape[j];
      j++;
    }
    result.push_back(data);
    i = j;
  }

  return result;
}

/// Compute the inverse of a permutation.
/// If perm[i] = j, then inverse[j] = i.
SmallVector<int64_t> inversePermutation(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> inverse(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    inverse[perm[i]] = i;
  }
  return inverse;
}

/// Check if two permutations are equal.
bool permutationsEqual(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  if (a.size() != b.size())
    return false;
  return std::equal(a.begin(), a.end(), b.begin());
}

/// Helper function to get permutation from transpose op
SmallVector<int64_t> getPermFromTranspose(mlir::ONNXTransposeOp transposeOp) {
  SmallVector<int64_t> perm;
  auto permAttr = transposeOp.getPermAttr();
  if (permAttr) {
    for (const auto &val : permAttr) {
      perm.push_back(mlir::cast<IntegerAttr>(val).getInt());
    }
  }
  return perm;
}

/// Pattern to match and optimize Dequantizelinear -> ONNXTranspose ->
/// quantizelinear -> dequantizelinear -> ONNXReshape -> quantizelinear ->
/// dequantizelinear -> ONNXTranspose -> quantizelinear sequences. This pattern
/// fires when consecutive dimensions can be merged, making the intermediate
/// operations redundant.
struct RemoveContinuousTransposeWithReshapePattern
    : public OpRewritePattern<mlir::ONNXTransposeOp> {
  using OpRewritePattern<mlir::ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXTransposeOp transpose1,
      PatternRewriter &rewriter) const override {
    DEBUG_WITH_TYPE("remove-continuous-transpose-with-reshape",
        llvm::errs() << "Trying to match " << transpose1 << "\n");

    // Get the input to transpose1 (should be a reshape)
    auto reshapeOp = transpose1.getData().getDefiningOp<ONNXReshapeOp>();
    if (!reshapeOp)
      return failure();

    // Get the input to reshape (should be transpose0)
    auto transpose0 = reshapeOp.getData().getDefiningOp<ONNXTransposeOp>();
    if (!transpose0)
      return failure();

    // Check that intermediate results have single use
    if (!reshapeOp.getResult().hasOneUse() ||
        !transpose0.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(
          transpose1, "Intermediate results do not have single use");

    // Extract permutation orders
    SmallVector<int64_t> order0 = getPermFromTranspose(transpose0);
    SmallVector<int64_t> order1 = getPermFromTranspose(transpose1);

    // Compute merged orders
    auto newOrder0 = computeNewOrder(order0);
    auto newOrder1 = computeNewOrder(order1);

    // Check if merged order1 equals the inverse of merged order0
    auto inverseNewOrder0 = inversePermutation(newOrder0);
    if (!permutationsEqual(newOrder1, inverseNewOrder0))
      return rewriter.notifyMatchFailure(
          transpose1, "Merged order1 is not the inverse of merged order0");

    auto transpose0OutputShapedType =
        mlir::cast<ShapedType>(transpose0.getResult().getType());
    auto reshapeOutputShapedType =
        mlir::cast<ShapedType>(reshapeOp.getResult().getType());
    // Ensure shapes are static
    if (!transpose0OutputShapedType.hasStaticShape() ||
        !reshapeOutputShapedType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          transpose1, "Transpose 0 output or Reshape output is not static");

    ArrayRef<int64_t> inputShape = transpose0OutputShapedType.getShape();
    ArrayRef<int64_t> outputShape = reshapeOutputShapedType.getShape();

    // Compute merged shapes
    auto newInputShape = computeNewShape(inputShape, order0);
    auto newOutputShape = computeNewShape(outputShape, order1);

    // Check if merged shapes are equal
    if (newInputShape != newOutputShape)
      return rewriter.notifyMatchFailure(transpose1,
          "Merged shapes are not equal, remove continuous reshape "
          "transpose optimization cannot be applied.");

    // The pattern matches! Now optimize.
    Value originalInput = transpose0.getData();
    auto originalInputShapedType =
        mlir::cast<ShapedType>(originalInput.getType());
    auto finalOutputShapedType =
        mlir::cast<ShapedType>(transpose1.getResult().getType());

    // Ensure original input has static shape
    if (!originalInputShapedType.hasStaticShape() ||
        !finalOutputShapedType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          transpose1, "Original input or final output is not static");

    // If shapes match, we can eliminate all three ops
    if (originalInputShapedType.getShape() ==
        finalOutputShapedType.getShape()) {
      // Direct replacement - no ops needed
      rewriter.replaceOp(transpose1, originalInput);
    } else {
      // Insert a single reshape to go from original input shape to final shape
      // Create the shape constant for the new reshape
      SmallVector<int64_t> newShapeValues(finalOutputShapedType.getShape());

      auto shapeType = RankedTensorType::get(
          {static_cast<int64_t>(newShapeValues.size())}, rewriter.getI64Type());

      auto shapeAttr = DenseElementsAttr::get(
          shapeType, llvm::ArrayRef<int64_t>(newShapeValues));

      // Use the simple 2-argument build: (sparse_value, value)
      auto shapeConstOp =
          rewriter.create<mlir::ONNXConstantOp>(transpose1.getLoc(),
              /*sparse_value=*/Attribute(),
              /*value=*/shapeAttr);

      auto newReshape = rewriter.create<mlir::ONNXReshapeOp>(
          transpose1.getLoc(), transpose1.getResult().getType(), originalInput,
          shapeConstOp.getResult());

      rewriter.replaceOp(transpose1, newReshape.getResult());
    }

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct RemoveContinuousTransposeWithReshapePass
    : public PassWrapper<RemoveContinuousTransposeWithReshapePass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "remove-continuous-transpose-with-reshape";
  }
  StringRef getDescription() const override {
    return "Remove redundant Transpose-Reshape-Transpose sequences when "
           "consecutive dimensions can be merged";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RemoveContinuousTransposeWithReshapePattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createRemoveContinuousTransposeWithReshapePass() {
  return std::make_unique<RemoveContinuousTransposeWithReshapePass>();
}

} // namespace onnx_mlir
