//===- ONNXTransposeOptimizationPatterns.h - Generic Transpose Optimization -*-
// C++ -*-===//
//
// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines generic patterns and infrastructure for optimizing
/// transposes with ONNX operations that have axis-dependent attributes.
///
/// NOTE: When a pattern replaces an operation, MLIR automatically keeps the
/// original transpose alive if it has other uses, enabling multi-use support.
///
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNXTRANSPOSEOPTIMIZATIONAXISCHANGEPATTERNS_H
#define ONNX_MLIR_ONNXTRANSPOSEOPTIMIZATIONAXISCHANGEPATTERNS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "onnx-transpose-optimization"

namespace mlir {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Compute inverse permutation
static SmallVector<int64_t> inversePermutation(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> inverse(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    inverse[perm[i]] = i;
  }
  return inverse;
}

//===----------------------------------------------------------------------===//
// AxisAttributeTransformer - Template for Attribute Transformation
//===----------------------------------------------------------------------===//

/// Generic template for transforming axis-dependent attributes when pushing
/// a transpose through an operation. Each operation that needs attribute
/// transformation should specialize this template.
///
/// The transformer provides two key methods:
/// 1. `transformAttributes` - Transforms operation attributes based on the
///    transpose permutation
/// 2. `getAdjustedPermutation` - Computes the adjusted permutation for
///    rank-changing operations (returns empty vector for non-rank-changing ops)
template <typename OpType>
struct AxisAttributeTransformer {
  /// Transform the operation's attributes based on the transpose permutation.
  /// Returns success() if transformation succeeds, failure() otherwise.
  static LogicalResult transformAttributes(OpType /*op*/,
      PatternRewriter & /*rewriter*/, ArrayRef<int64_t> /*perm*/) {
    // Default: no transformation needed
    return success();
  }

  /// Compute the adjusted permutation for rank-changing operations.
  /// For operations that change the rank (e.g., Squeeze, Reduce with
  /// keepdims=0), this method computes the new permutation to apply after the
  /// operation. Returns an empty vector for operations that don't change rank.
  static SmallVector<int64_t> getAdjustedPermutation(OpType /*op*/,
      ArrayRef<int64_t> /*perm*/, ArrayRef<int64_t> /*inputShape*/,
      ArrayRef<int64_t> /*outputShape*/) {
    // Default: no rank change, return empty vector
    return SmallVector<int64_t>();
  }
};

//===----------------------------------------------------------------------===//
// Helper Functions for Attribute Transformation
//===----------------------------------------------------------------------===//

/// Helper function to transform reduction operation attributes
LogicalResult transformReductionAttributes(
    Operation *op, PatternRewriter &rewriter, ArrayRef<int64_t> perm);

/// Helper function to compute adjusted permutation for reduction operations
SmallVector<int64_t> getReductionAdjustedPermutation(Operation *op,
    ArrayRef<int64_t> perm, ArrayRef<int64_t> inputShape,
    ArrayRef<int64_t> outputShape);

//===----------------------------------------------------------------------===//
// AxisAttributeTransformer Specializations
//===----------------------------------------------------------------------===//

// ONNXPadOp - Transform pads attribute
template <>
struct AxisAttributeTransformer<ONNXPadOp> {
  static LogicalResult transformAttributes(
      ONNXPadOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm);

  static SmallVector<int64_t> getAdjustedPermutation(ONNXPadOp /*op*/,
      ArrayRef<int64_t> /*perm*/, ArrayRef<int64_t> /*inputShape*/,
      ArrayRef<int64_t> /*outputShape*/) {
    return SmallVector<int64_t>(); // No rank change
  }
};

// ONNXSliceOp - Transform starts, ends, axes, steps attributes
template <>
struct AxisAttributeTransformer<ONNXSliceOp> {
  static LogicalResult transformAttributes(
      ONNXSliceOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm);

  static SmallVector<int64_t> getAdjustedPermutation(ONNXSliceOp /*op*/,
      ArrayRef<int64_t> /*perm*/, ArrayRef<int64_t> /*inputShape*/,
      ArrayRef<int64_t> /*outputShape*/) {
    return SmallVector<int64_t>(); // No rank change
  }
};

// ONNXExpandOp - Transform shape attribute
template <>
struct AxisAttributeTransformer<ONNXExpandOp> {
  static LogicalResult transformAttributes(
      ONNXExpandOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm);

  static SmallVector<int64_t> getAdjustedPermutation(ONNXExpandOp /*op*/,
      ArrayRef<int64_t> /*perm*/, ArrayRef<int64_t> /*inputShape*/,
      ArrayRef<int64_t> /*outputShape*/) {
    return SmallVector<int64_t>(); // No rank change
  }
};

// ONNXTileOp - Transform repeats attribute
template <>
struct AxisAttributeTransformer<ONNXTileOp> {
  static LogicalResult transformAttributes(
      ONNXTileOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm);

  static SmallVector<int64_t> getAdjustedPermutation(ONNXTileOp /*op*/,
      ArrayRef<int64_t> /*perm*/, ArrayRef<int64_t> /*inputShape*/,
      ArrayRef<int64_t> /*outputShape*/) {
    return SmallVector<int64_t>(); // No rank change
  }
};

// ONNXSqueezeOp - Transform axes attribute and compute adjusted permutation
template <>
struct AxisAttributeTransformer<ONNXSqueezeOp> {
  static LogicalResult transformAttributes(
      ONNXSqueezeOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm);

  static SmallVector<int64_t> getAdjustedPermutation(ONNXSqueezeOp op,
      ArrayRef<int64_t> perm, ArrayRef<int64_t> inputShape,
      ArrayRef<int64_t> outputShape);
};

// ONNXArgMaxOp - Transform axis attribute and compute adjusted permutation
template <>
struct AxisAttributeTransformer<ONNXArgMaxOp> {
  static LogicalResult transformAttributes(
      ONNXArgMaxOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm);

  static SmallVector<int64_t> getAdjustedPermutation(ONNXArgMaxOp op,
      ArrayRef<int64_t> perm, ArrayRef<int64_t> inputShape,
      ArrayRef<int64_t> outputShape);
};

// ONNXSoftmaxOp - Transform axis attribute
template <>
struct AxisAttributeTransformer<ONNXSoftmaxOp> {
  static LogicalResult transformAttributes(
      ONNXSoftmaxOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
    int64_t axis = op.getAxis();
    int64_t rank = static_cast<int64_t>(perm.size());

    if (axis < 0)
      axis += rank;

    if (axis < 0 || axis >= rank)
      return failure();

    int64_t newAxis = perm[axis];

    op.setAxisAttr(rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, /*isSigned=*/true), newAxis));

    return success();
  }

  static SmallVector<int64_t> getAdjustedPermutation(ONNXSoftmaxOp /*op*/,
      ArrayRef<int64_t> /*perm*/, ArrayRef<int64_t> /*inputShape*/,
      ArrayRef<int64_t> /*outputShape*/) {
    return SmallVector<int64_t>(); // No rank change
  }
};

// Reduction Operations - Transform axes attribute and compute adjusted
// permutation
template <>
struct AxisAttributeTransformer<ONNXReduceMeanOp> {
  static LogicalResult transformAttributes(
      ONNXReduceMeanOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
    return transformReductionAttributes(op.getOperation(), rewriter, perm);
  }

  static SmallVector<int64_t> getAdjustedPermutation(ONNXReduceMeanOp op,
      ArrayRef<int64_t> perm, ArrayRef<int64_t> inputShape,
      ArrayRef<int64_t> outputShape) {
    return getReductionAdjustedPermutation(
        op.getOperation(), perm, inputShape, outputShape);
  }
};

template <>
struct AxisAttributeTransformer<ONNXReduceMaxOp> {
  static LogicalResult transformAttributes(
      ONNXReduceMaxOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
    return transformReductionAttributes(op.getOperation(), rewriter, perm);
  }

  static SmallVector<int64_t> getAdjustedPermutation(ONNXReduceMaxOp op,
      ArrayRef<int64_t> perm, ArrayRef<int64_t> inputShape,
      ArrayRef<int64_t> outputShape) {
    return getReductionAdjustedPermutation(
        op.getOperation(), perm, inputShape, outputShape);
  }
};

template <>
struct AxisAttributeTransformer<ONNXReduceMinOp> {
  static LogicalResult transformAttributes(
      ONNXReduceMinOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
    return transformReductionAttributes(op.getOperation(), rewriter, perm);
  }

  static SmallVector<int64_t> getAdjustedPermutation(ONNXReduceMinOp op,
      ArrayRef<int64_t> perm, ArrayRef<int64_t> inputShape,
      ArrayRef<int64_t> outputShape) {
    return getReductionAdjustedPermutation(
        op.getOperation(), perm, inputShape, outputShape);
  }
};

template <>
struct AxisAttributeTransformer<ONNXReduceSumOp> {
  static LogicalResult transformAttributes(
      ONNXReduceSumOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
    return transformReductionAttributes(op.getOperation(), rewriter, perm);
  }

  static SmallVector<int64_t> getAdjustedPermutation(ONNXReduceSumOp op,
      ArrayRef<int64_t> perm, ArrayRef<int64_t> inputShape,
      ArrayRef<int64_t> outputShape) {
    return getReductionAdjustedPermutation(
        op.getOperation(), perm, inputShape, outputShape);
  }
};

//===----------------------------------------------------------------------===//
// PushTransposeThroughAxisOp - Generic Pattern Template
//===----------------------------------------------------------------------===//

/// Generic pattern for pushing transpose through operations with axis-dependent
/// attributes. This pattern:
/// 1. Matches a transpose followed by an operation
/// 2. Transforms the operation's attributes using AxisAttributeTransformer
/// 3. Creates a new operation with transformed attributes
/// 5. Pushes the transpose after the operation
/// 6. Handles rank-changing operations by computing adjusted permutation
template <typename OpType>
struct PushTransposeThroughAxisOp : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      OpType op, PatternRewriter &rewriter) const override {
    // Get the first input operand (data input)
    Value input;
    if constexpr (std::is_same_v<OpType, ONNXSqueezeOp> ||
                  std::is_same_v<OpType, ONNXArgMaxOp>) {
      input = op.getData();
    } else if constexpr (std::is_same_v<OpType, ONNXSoftmaxOp>) {
      input = op.getInput();
    } else {
      input = op.getOperand(0);
    }

    // Check if input comes from a transpose
    auto transposeOp = input.getDefiningOp<ONNXTransposeOp>();
    if (!transposeOp)
      return failure();

    // Get transpose permutation
    auto permAttr = transposeOp.getPermAttr();
    if (!permAttr)
      return failure();

    SmallVector<int64_t> perm;
    for (auto attr : permAttr.getValue())
      perm.push_back(mlir::cast<IntegerAttr>(attr).getInt());

    LLVM_DEBUG(llvm::dbgs() << "Pushing transpose through axis-dependent op "
                            << op->getName().getStringRef() << "\n");

    // Get input and output shapes for rank-change detection
    auto inputType =
        mlir::cast<RankedTensorType>(transposeOp.getData().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto inputShape = inputType.getShape();

    // Create a copy of the operation with the original (non-transposed) input
    rewriter.setInsertionPoint(op);
    auto newOp = mlir::cast<OpType>(rewriter.clone(*op.getOperation()));

    // The cloned intermediate must not carry ResultNames, it will be inferred
    // through transpose
    newOp->removeAttr("ResultNames");

    // Update the first operand to use the transpose's input
    if constexpr (std::is_same_v<OpType, ONNXSqueezeOp> ||
                  std::is_same_v<OpType, ONNXArgMaxOp>) {
      newOp.getDataMutable().assign(transposeOp.getData());
    } else if constexpr (std::is_same_v<OpType, ONNXSoftmaxOp>) {
      newOp.getInputMutable().assign(transposeOp.getData());
    } else {
      newOp.setOperand(0, transposeOp.getData());
    }

    // Transform attributes using the specialized transformer
    if (failed(AxisAttributeTransformer<OpType>::transformAttributes(
            newOp, rewriter, perm))) {
      rewriter.eraseOp(newOp);
      return failure();
    }

    // The copied operation has the wrong output type (from transposed space).
    // Set it to UnrankedTensorType and let shape inference compute the correct
    // type.
    auto elementType = mlir::cast<RankedTensorType>(newOp.getResult().getType())
                           .getElementType();
    newOp.getResult().setType(UnrankedTensorType::get(elementType));

    // Trigger shape inference to compute the correct output type
    if (failed(newOp.inferShapes([](Region &region) {}))) {
      // Shape inference failed - cannot transform this operation
      rewriter.eraseOp(newOp);
      return failure();
    }

    // Get the new operation's output type (after shape inference)
    auto newOpOutputType =
        mlir::dyn_cast<RankedTensorType>(newOp.getResult().getType());
    if (!newOpOutputType) {
      rewriter.eraseOp(newOp);
      return failure();
    }
    auto newOpOutputShape = newOpOutputType.getShape();

    // Check if rank changed
    bool rankChanged = (inputShape.size() != newOpOutputShape.size());

    // Compute the permutation for the output transpose
    SmallVector<int64_t> newPerm;
    if (rankChanged) {
      // Get adjusted permutation for rank-changing operation
      newPerm = AxisAttributeTransformer<OpType>::getAdjustedPermutation(
          newOp, perm, inputShape, newOpOutputShape);

      // If no adjusted permutation returned, transformation failed
      if (newPerm.empty()) {
        rewriter.eraseOp(newOp);
        return failure();
      }
    } else {
      // No rank change, use original permutation
      newPerm = perm;
    }

    // The final transpose should produce the exact same output as the original
    // operation Use the original output type directly
    auto newTransposeOp = rewriter.create<ONNXTransposeOp>(op.getLoc(),
        outputType, // Use original output type - transformation is semantically
                    // equivalent
        newOp.getResult(), rewriter.getI64ArrayAttr(newPerm));

    // Replace the original operation with the new transpose
    rewriter.replaceOp(op, newTransposeOp.getResult());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// PushTransposeThroughConcat - Specialized Pattern for Concat
//===----------------------------------------------------------------------===//

/// Specialized pattern for pushing transposes through Concat operations.
/// This pattern handles the case where ALL inputs to Concat come from
/// transposes with the SAME permutation. It:
/// 1. Verifies all inputs are from transposes with identical permutations
/// 2. Transforms the concat axis attribute
/// 3. Pushes a single transpose after the concat
struct PushTransposeThroughConcat : public OpRewritePattern<ONNXConcatOp> {
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const override {
    // Get all input operands
    auto inputs = concatOp.getInputs();
    if (inputs.empty())
      return failure();

    // Check that ALL inputs come from transpose operations
    SmallVector<ONNXTransposeOp> transposeOps;
    SmallVector<int64_t> firstPerm;

    for (auto input : inputs) {
      auto transposeOp = input.getDefiningOp<ONNXTransposeOp>();
      if (!transposeOp)
        return failure();

      // Get transpose permutation
      auto permAttr = transposeOp.getPermAttr();
      if (!permAttr)
        return failure();

      SmallVector<int64_t> perm;
      for (auto attr : permAttr.getValue())
        perm.push_back(mlir::cast<IntegerAttr>(attr).getInt());

      // First transpose sets the expected permutation
      if (transposeOps.empty()) {
        firstPerm = perm;
      } else {
        // All subsequent transposes must have the same permutation
        if (perm != firstPerm)
          return failure();
      }

      transposeOps.push_back(transposeOp);
    }

    LLVM_DEBUG(llvm::dbgs() << "Pushing transposes through Concat\n");

    // Get the axis attribute
    auto axisAttr = concatOp.getAxisAttr();
    if (!axisAttr)
      return failure();

    int64_t axis = axisAttr.getValue().getSExtValue();
    auto rank = static_cast<int64_t>(firstPerm.size());

    // Normalize negative axis
    if (axis < 0)
      axis += rank;

    if (axis < 0 || axis >= rank)
      return failure();

    // Transform axis: axis in transposed space corresponds to firstPerm[axis]
    // in original space
    int64_t newAxis = firstPerm[axis];

    // Create new concat with original (non-transposed) inputs
    SmallVector<Value> newInputs;
    for (auto transposeOp : transposeOps) {
      newInputs.push_back(transposeOp.getData());
    }

    // Get the original concat output type before transpose (must be ranked)
    auto originalConcatOutputType =
        mlir::dyn_cast<RankedTensorType>(concatOp.getType());
    if (!originalConcatOutputType)
      return failure(); // Unranked tensor - cannot optimize

    // Compute the new concat output shape (in original space)
    SmallVector<int64_t> newConcatShape;
    auto firstInputType =
        mlir::dyn_cast<RankedTensorType>(transposeOps[0].getData().getType());
    if (!firstInputType)
      return failure(); // Unranked tensor - cannot optimize
    newConcatShape.assign(
        firstInputType.getShape().begin(), firstInputType.getShape().end());

    // Compute concatenated dimension size
    int64_t concatDimSize = 0;
    for (auto transposeOp : transposeOps) {
      auto inputType =
          mlir::dyn_cast<RankedTensorType>(transposeOp.getData().getType());
      if (!inputType)
        return failure(); // Unranked tensor - cannot optimize
      auto inputShape = inputType.getShape();
      if (newAxis >= 0 && static_cast<size_t>(newAxis) < inputShape.size()) {
        if (concatDimSize == 0)
          concatDimSize = inputShape[newAxis];
        else
          concatDimSize += inputShape[newAxis];
      }
    }
    newConcatShape[newAxis] = concatDimSize;

    auto newConcatType = RankedTensorType::get(
        newConcatShape, originalConcatOutputType.getElementType());

    // Create new axis attribute (signed integer)
    auto newAxisAttr = rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, /*isSigned=*/true), newAxis);

    // Create new concat operation
    rewriter.setInsertionPoint(concatOp);
    auto newConcatOp = rewriter.create<ONNXConcatOp>(
        concatOp.getLoc(), newConcatType, newInputs, newAxisAttr);

    // Create transpose after concat
    auto newTransposeOp = rewriter.create<ONNXTransposeOp>(concatOp.getLoc(),
        originalConcatOutputType, newConcatOp.getResult(),
        rewriter.getI64ArrayAttr(firstPerm));

    // Replace the original concat with the new transpose
    rewriter.replaceOp(concatOp, newTransposeOp.getResult());

    return success();
  }
};

} // namespace mlir

#endif // ONNX_MLIR_ONNXTRANSPOSEOPTIMIZATIONAXISCHANGEPATTERNS_H
