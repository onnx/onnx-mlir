// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass optimizes the Slice-Reshape-Transpose block for MHA (Multi-Head
// Attention) patterns. Transforms:
//   input -> [slice_0, slice_1, slice_2] -> [reshape_0, reshape_1, reshape_2]
//          -> [transpose_0, transpose_1, transpose_2] -> consumer
// Into:
//   input -> reshape -> transpose -> [slice_0, slice_1, slice_2] -> consumer

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "optimize-slice-reshape-transpose-block"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Helper to get shape from a ranked tensor type
SmallVector<int64_t> getShapeFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    auto shape = tensorType.getShape();
    return SmallVector<int64_t>(shape.begin(), shape.end());
  }
  return {};
}

/// Helper to get element type from a tensor type
Type getElementType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return tensorType.getElementType();
  }
  return nullptr;
}

/// Create a constant op with the given int64 values
Value createConstantI64Array(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> values) {
  MLIRContext *ctx = rewriter.getContext();
  auto tensorType = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, IntegerType::get(ctx, 64));
  auto denseAttr = DenseIntElementsAttr::get(tensorType, values);
  return rewriter.create<ONNXConstantOp>(loc, Attribute(), denseAttr);
}

/// Check if the transpose order matches expected pattern
bool isExpectedTransposeOrder(
    ArrayAttr permAttr, ArrayRef<int64_t> expectedOrder) {
  if (!permAttr || permAttr.size() != expectedOrder.size())
    return false;

  for (size_t i = 0; i < expectedOrder.size(); ++i) {
    if (cast<IntegerAttr>(permAttr[i]).getInt() != expectedOrder[i])
      return false;
  }
  return true;
}

/// Structure to hold the matched pattern
struct MatchedPattern {
  ONNXSliceOp slices[3];
  ONNXReshapeOp reshapes[3];
  ONNXTransposeOp transposes[3];
  Value commonInput;
  SmallVector<int64_t> reshapeShape;
  SmallVector<int64_t> inputShape;
};

/// Try to match the MHA Slice-Reshape-Transpose pattern starting from a
/// transpose
LogicalResult matchPattern(
    ONNXTransposeOp transposeOp, MatchedPattern &pattern) {
  // Check transpose order - must be {0, 2, 1, 3}
  auto permAttr = transposeOp.getPermAttr();
  if (!isExpectedTransposeOrder(permAttr, {0, 2, 1, 3})) {
    return failure();
  }

  // Get the reshape op feeding this transpose
  auto reshapeOp = transposeOp.getData().getDefiningOp<ONNXReshapeOp>();
  if (!reshapeOp || !reshapeOp.getResult().hasOneUse()) {
    return failure();
  }

  // Get the slice op feeding the reshape
  auto sliceOp = reshapeOp.getData().getDefiningOp<ONNXSliceOp>();
  if (!sliceOp || !sliceOp.getResult().hasOneUse()) {
    return failure();
  }

  // Get the common input to the slice
  Value sliceInput = sliceOp.getData();

  // Now find all sibling slices that share the same input
  SmallVector<ONNXSliceOp> siblingSlices;
  for (Operation *user : sliceInput.getUsers()) {
    if (auto sibSlice = dyn_cast<ONNXSliceOp>(user)) {
      siblingSlices.push_back(sibSlice);
    }
  }

  // We need exactly 3 slices for the MHA pattern
  if (siblingSlices.size() != 3) {
    return failure();
  }

  // Verify each slice -> reshape -> transpose chain has correct structure
  int validChains = 0;
  for (auto sibSlice : siblingSlices) {
    // Check single use to reshape
    if (!sibSlice.getResult().hasOneUse())
      continue;

    auto sibReshape =
        dyn_cast<ONNXReshapeOp>(*sibSlice.getResult().getUsers().begin());
    if (!sibReshape || !sibReshape.getResult().hasOneUse())
      continue;

    auto sibTranspose =
        dyn_cast<ONNXTransposeOp>(*sibReshape.getResult().getUsers().begin());
    if (!sibTranspose)
      continue;

    // Check transpose order is {0, 2, 1, 3}
    if (!isExpectedTransposeOrder(sibTranspose.getPermAttr(), {0, 2, 1, 3}))
      continue;

    // Store in pattern
    pattern.slices[validChains] = sibSlice;
    pattern.reshapes[validChains] = sibReshape;
    pattern.transposes[validChains] = sibTranspose;
    validChains++;
  }

  // Store common input and shapes
  pattern.commonInput = sliceInput;
  pattern.inputShape = getShapeFromType(sliceInput.getType());
  pattern.reshapeShape =
      getShapeFromType(pattern.reshapes[0].getResult().getType());

  // Verify input shape is 3D and reshape shape is 4D
  if (pattern.inputShape.size() != 3 || pattern.reshapeShape.size() != 4) {
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pattern: Optimize MHA Slice-Reshape-Transpose Block
//===----------------------------------------------------------------------===//

class OptimizeSliceReshapeTransposeMHAPattern
    : public OpRewritePattern<ONNXTransposeOp> {
public:
  using OpRewritePattern<ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXTransposeOp transposeOp, PatternRewriter &rewriter) const override {
    // Try to match the pattern
    MatchedPattern pattern;
    if (failed(matchPattern(transposeOp, pattern))) {
      return failure();
    }

    Location loc = transposeOp.getLoc();

    // Get element type from input - preserve it throughout
    Type inputType = pattern.commonInput.getType();
    Type elementType = getElementType(inputType);
    if (!elementType) {
      return failure();
    }

    // Calculate new reshape shape: {N, S, num_heads * 3, head_dim}
    // Original: input {N, S, C} -> slice -> reshape {N, S, num_heads, head_dim}
    // New: input {N, S, C} -> reshape {N, S, C/head_dim, head_dim} -> transpose
    auto &inputShape = pattern.inputShape;
    auto &reshapeShape = pattern.reshapeShape;

    int64_t headDim = reshapeShape[3];
    SmallVector<int64_t> newReshapeShape = {
        inputShape[0], inputShape[1], inputShape[2] / headDim, headDim};

    // Set insertion point before the first slice
    rewriter.setInsertionPoint(pattern.slices[0]);

    // Create the new reshape output type preserving element type
    auto newReshapeType = RankedTensorType::get(newReshapeShape, elementType);

    // Create reshape shape constant
    Value newReshapeShapeConst =
        createConstantI64Array(rewriter, loc, newReshapeShape);

    // Create the new reshape op
    auto newReshapeOp = rewriter.create<ONNXReshapeOp>(
        loc, newReshapeType, pattern.commonInput, newReshapeShapeConst);

    // Create new transpose with order {0, 2, 1, 3}
    SmallVector<int64_t> newTransposeShape = {newReshapeShape[0],
        newReshapeShape[2], newReshapeShape[1], newReshapeShape[3]};
    auto newTransposeType =
        RankedTensorType::get(newTransposeShape, elementType);

    auto newTransposeOp =
        rewriter.create<ONNXTransposeOp>(loc, newTransposeType,
            newReshapeOp.getResult(), rewriter.getI64ArrayAttr({0, 2, 1, 3}));

    // Calculate slice parameters for the 3 slices
    // Assuming slices divide the second dimension (num_heads dimension after
    // transpose)
    int64_t numHeadsPerSlice = newTransposeShape[1] / 3;

    // Create new slices and collect replacement values
    SmallVector<Value> newSliceResults;
    for (int i = 0; i < 3; i++) {
      // New slice parameters on the transposed tensor
      SmallVector<int64_t> newStarts = {0, i * numHeadsPerSlice, 0, 0};
      SmallVector<int64_t> newEnds = {newTransposeShape[0],
          (i + 1) * numHeadsPerSlice, newTransposeShape[2],
          newTransposeShape[3]};
      SmallVector<int64_t> newSteps = {1, 1, 1, 1};
      SmallVector<int64_t> newAxes = {0, 1, 2, 3};

      // Create constants for slice parameters
      Value startsConst = createConstantI64Array(rewriter, loc, newStarts);
      Value endsConst = createConstantI64Array(rewriter, loc, newEnds);
      Value stepsConst = createConstantI64Array(rewriter, loc, newSteps);
      Value axesConst = createConstantI64Array(rewriter, loc, newAxes);

      // Calculate new slice output shape preserving element type
      SmallVector<int64_t> newSliceShape = {newTransposeShape[0],
          numHeadsPerSlice, newTransposeShape[2], newTransposeShape[3]};
      auto newSliceType = RankedTensorType::get(newSliceShape, elementType);

      // Create new slice op
      auto newSliceOp = rewriter.create<ONNXSliceOp>(loc, newSliceType,
          newTransposeOp.getResult(), startsConst, endsConst, axesConst,
          stepsConst);

      newSliceResults.push_back(newSliceOp.getResult());
    }

    // Replace uses and erase old ops
    for (int i = 0; i < 3; i++) {
      rewriter.replaceOp(pattern.transposes[i], newSliceResults[i]);
    }

    // Erase intermediate ops (reshape and slice) - they should now be unused
    for (int i = 0; i < 3; i++) {
      if (pattern.reshapes[i]->use_empty())
        rewriter.eraseOp(pattern.reshapes[i]);
      if (pattern.slices[i]->use_empty())
        rewriter.eraseOp(pattern.slices[i]);
    }

    return success();
  }
};

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass to optimize MHA Slice-Reshape-Transpose blocks
struct OptimizeSliceReshapeTransposeBlockPass
    : public PassWrapper<OptimizeSliceReshapeTransposeBlockPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "optimize-slice-reshape-transpose-block";
  }
  StringRef getDescription() const override {
    return "Optimize MHA Slice-Reshape-Transpose blocks by moving reshape and "
           "transpose before slices";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<OptimizeSliceReshapeTransposeMHAPattern>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createOptimizeSliceReshapeTransposeBlockPass() {
  return std::make_unique<OptimizeSliceReshapeTransposeBlockPass>();
}

} // namespace onnx_mlir
