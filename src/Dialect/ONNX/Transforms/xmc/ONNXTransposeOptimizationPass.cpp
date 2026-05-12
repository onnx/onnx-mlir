//===- ONNXTransposeOptimizationPass.cpp - Optimize ONNX Transpose Operations
//-------===//
//
// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass implements comprehensive transpose fusion optimizations based on
// the vaip_pass_fuse_transpose pass, covering all 66+ transformation rules.
//
//===----------------------------------------------------------------------===//

#include "ONNXTransposeOptimizationAxisChangePatterns.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <numeric>
#include <optional>

#define DEBUG_TYPE "onnx-transpose-optimization"

using namespace mlir;

namespace onnx_mlir {

namespace {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Remap per-axis quantization dimension through a transpose permutation.
/// For perm[i] == oldAxis, the new axis becomes i.
static Type remapPerAxisQuantType(Type elementType, ArrayRef<int64_t> perm) {
  auto perAxisType = dyn_cast<quant::UniformQuantizedPerAxisType>(elementType);
  if (!perAxisType)
    return elementType;

  int32_t oldAxis = perAxisType.getQuantizedDimension();
  int32_t newAxis = oldAxis;
  for (int64_t i = 0, e = perm.size(); i < e; ++i) {
    if (perm[i] == oldAxis) {
      newAxis = static_cast<int32_t>(i);
      break;
    }
  }
  if (newAxis == oldAxis)
    return elementType;

  return quant::UniformQuantizedPerAxisType::get(perAxisType.getFlags(),
      perAxisType.getStorageType(), perAxisType.getExpressedType(),
      perAxisType.getScales(), perAxisType.getZeroPoints(), newAxis,
      perAxisType.getStorageTypeMin(), perAxisType.getStorageTypeMax());
}

/// Check if permutation is identity [0, 1, 2, ..., n-1]
bool isIdentityPermutation(ArrayRef<int64_t> perm) {
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i))
      return false;
  }
  return true;
}

/// Compose two permutations: result[i] = perm1[perm2[i]]
SmallVector<int64_t> composePermutations(
    ArrayRef<int64_t> perm1, ArrayRef<int64_t> perm2) {
  assert(perm1.size() == perm2.size() && "Permutation size mismatch");
  SmallVector<int64_t> result(perm1.size());
  for (size_t i = 0; i < perm1.size(); ++i) {
    result[i] = perm1[perm2[i]];
  }
  return result;
}

/// Apply permutation to shape: result[i] = shape[perm[i]].
/// If shape has fewer dimensions than perm, it is left-padded with 1s
/// (matching ONNX right-aligned broadcast semantics) before permuting.
SmallVector<int64_t> permuteShape(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> perm) {
  ArrayRef<int64_t> workShape = shape;
  SmallVector<int64_t> expanded;
  if (shape.size() < perm.size()) {
    // Left-pad with 1s to match perm rank.
    expanded.resize(perm.size() - shape.size(), 1);
    expanded.append(shape.begin(), shape.end());
    workShape = expanded;
  }
  assert(perm.size() == workShape.size() &&
         "Permutation size must match shape size");
  for (int64_t idx : perm) {
    assert(idx >= 0 && static_cast<size_t>(idx) < workShape.size() &&
           "Permutation index out of bounds");
  }

  SmallVector<int64_t> result(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    result[i] = workShape[perm[i]];
  }
  return result;
}

/// Check if shape is transpose-immune (scalar or only 1 non-unit dimension)
bool isTransposeImmune(ArrayRef<int64_t> shape) {
  if (shape.empty() || shape.size() == 1)
    return true;

  int nonOnes = 0;
  for (auto dim : shape) {
    if (dim != 1)
      ++nonOnes;
  }
  return nonOnes <= 1;
}

/// Expand shape by prepending 1s to match target size
/// Used for rank-mismatch handling in broadcast operations
SmallVector<int64_t> expandShape(ArrayRef<int64_t> shape, size_t targetSize) {
  SmallVector<int64_t> expandedShape;
  assert(shape.size() <= targetSize && "Shape size must be <= target size");

  // Prepend 1s to match target size
  if (shape.size() < targetSize) {
    expandedShape.append(targetSize - shape.size(), 1);
  }

  // Append original shape
  expandedShape.append(shape.begin(), shape.end());

  return expandedShape;
}

/// Get permutation from ONNX Transpose operation
std::optional<SmallVector<int64_t>> getTransposePermutation(
    ONNXTransposeOp transposeOp) {
  auto permAttr = transposeOp.getPermAttr();
  if (!permAttr)
    return std::nullopt;

  SmallVector<int64_t> perm;
  for (auto attr : permAttr.getValue())
    perm.push_back(mlir::cast<IntegerAttr>(attr).getInt());

  return perm;
}

//===----------------------------------------------------------------------===//
// Pattern 1: Eliminate Identity Transpose
//===----------------------------------------------------------------------===//

struct EliminateIdentityTranspose : public OpRewritePattern<ONNXTransposeOp> {
  using OpRewritePattern<ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXTransposeOp op, PatternRewriter &rewriter) const override {
    auto perm = getTransposePermutation(op);
    if (!perm || !isIdentityPermutation(*perm))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Eliminating identity transpose\n");
    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 2: Fuse Consecutive Transposes
// Rule: transpose(transpose(x, perm1), perm2) -> transpose(x, perm1∘perm2)
//===----------------------------------------------------------------------===//

struct FuseConsecutiveTransposes : public OpRewritePattern<ONNXTransposeOp> {
  using OpRewritePattern<ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXTransposeOp op, PatternRewriter &rewriter) const override {
    auto prevTranspose = op.getOperand().getDefiningOp<ONNXTransposeOp>();
    if (!prevTranspose)
      return failure();

    auto perm1 = getTransposePermutation(prevTranspose);
    auto perm2 = getTransposePermutation(op);

    if (!perm1 || !perm2)
      return failure();

    if (perm1->size() != perm2->size())
      return failure();

    auto composedPerm = composePermutations(*perm1, *perm2);

    LLVM_DEBUG(llvm::dbgs() << "Fusing consecutive transposes\n");

    if (isIdentityPermutation(composedPerm)) {
      auto inputType = mlir::dyn_cast<RankedTensorType>(
          prevTranspose.getOperand().getType());
      auto outputType = mlir::dyn_cast<RankedTensorType>(op.getType());

      // If transposes compose to identity but change element type, the
      // QuantTypesPass embedded a Q or DQ into the transpose output type. We
      // need to re-materialize the Q/DQ operation to preserve semantics without
      // transpose overhead.
      if (inputType && outputType &&
          inputType.getElementType() != outputType.getElementType()) {

        auto inputQuantType = mlir::dyn_cast<quant::UniformQuantizedType>(
            inputType.getElementType());
        auto outputQuantType = mlir::dyn_cast<quant::UniformQuantizedType>(
            outputType.getElementType());

        // Case 1: f32 -> quantized (re-materialize QuantizeLinear)
        if (!inputQuantType && outputQuantType) {
          LLVM_DEBUG(llvm::dbgs() << "  Folding transposes with quantization: "
                                  << inputType.getElementType() << " -> "
                                  << outputType.getElementType() << "\n");

          double scale = outputQuantType.getScale();
          int64_t zeroPoint = outputQuantType.getZeroPoint();

          // Create scale constant
          auto scaleTensorType =
              RankedTensorType::get({}, rewriter.getF32Type());
          auto scaleAttr = DenseElementsAttr::get(
              scaleTensorType, rewriter.getF32FloatAttr(scale));
          auto scaleConst = rewriter.create<ONNXConstantOp>(
              op.getLoc(), Attribute(), scaleAttr);

          // Create zero-point constant
          auto zpStorageType = outputQuantType.getStorageType();
          auto zpTensorType = RankedTensorType::get({}, zpStorageType);
          auto zpAttr = DenseElementsAttr::get(
              zpTensorType, rewriter.getIntegerAttr(zpStorageType, zeroPoint));
          auto zpConst =
              rewriter.create<ONNXConstantOp>(op.getLoc(), Attribute(), zpAttr);

          // QuantizeLinear produces storage type (i8), not quant type.
          auto storageResultType = RankedTensorType::get(
              outputType.getShape(), outputQuantType.getStorageType());
          auto qOp = rewriter.create<ONNXQuantizeLinearOp>(op.getLoc(),
              storageResultType, prevTranspose.getOperand(),
              scaleConst.getResult(), zpConst.getResult(),
              /*axis=*/IntegerAttr(), /*saturate=*/IntegerAttr(),
              /*block_size=*/IntegerAttr());

          // scast restores the quant type for downstream consumers.
          auto scastOp = rewriter.create<quant::StorageCastOp>(
              op.getLoc(), outputType, qOp.getResult());
          rewriter.replaceOp(op, scastOp.getResult());

          return success();
        }

        // Case 2: quantized -> f32 (re-materialize DequantizeLinear)
        // The type change means QuantTypesPass already embedded DQ into the
        // transpose. We just re-materialize it explicitly while folding the
        // transpose overhead.
        if (inputQuantType && !outputQuantType) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  Folding transposes with dequantization: "
                     << inputType.getElementType() << " -> "
                     << outputType.getElementType() << "\n");

          double scale = inputQuantType.getScale();
          int64_t zeroPoint = inputQuantType.getZeroPoint();

          // Create scale constant
          auto scaleTensorType =
              RankedTensorType::get({}, rewriter.getF32Type());
          auto scaleAttr = DenseElementsAttr::get(
              scaleTensorType, rewriter.getF32FloatAttr(scale));
          auto scaleConst = rewriter.create<ONNXConstantOp>(
              op.getLoc(), Attribute(), scaleAttr);

          // Create zero-point constant
          auto zpStorageType = inputQuantType.getStorageType();
          auto zpTensorType = RankedTensorType::get({}, zpStorageType);
          auto zpAttr = DenseElementsAttr::get(
              zpTensorType, rewriter.getIntegerAttr(zpStorageType, zeroPoint));
          auto zpConst =
              rewriter.create<ONNXConstantOp>(op.getLoc(), Attribute(), zpAttr);

          // scast strips the quant type so DQ gets plain storage type input.
          Value dqInput = prevTranspose.getOperand();
          auto storageTy = RankedTensorType::get(
              inputType.getShape(), inputQuantType.getStorageType());
          auto scastOp = rewriter.create<quant::StorageCastOp>(
              op.getLoc(), storageTy, dqInput);

          // DequantizeLinear takes storage type (i8) input, produces f32.
          rewriter.replaceOpWithNewOp<ONNXDequantizeLinearOp>(op, outputType,
              scastOp.getResult(), scaleConst.getResult(), zpConst.getResult(),
              /*axis=*/IntegerAttr(), /*block_size=*/IntegerAttr());

          return success();
        }

        // Other type changes (not quantization-related), skip optimization
        return failure();
      }

      rewriter.replaceOp(op, prevTranspose.getOperand());
    } else {
      rewriter.replaceOpWithNewOp<ONNXTransposeOp>(op, op.getType(),
          prevTranspose.getOperand(), rewriter.getI64ArrayAttr(composedPerm));
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 2d: Push Transpose Through Reshape
// Rule: transpose(reshape(x)) -> reshape(transpose(x))
//===----------------------------------------------------------------------===//

// Pattern: Move Transpose Through Reshape
// Transforms: Transpose → Reshape into Reshape → Transpose
// This is ONLY safe when the reshape operation is "factorizable":
// - It only splits/merges dimensions independently
// - No data mixing across dimensions occurs
// Example: [N,C,H,W] → Transpose[0,2,3,1] → [N,H,W,C] → Reshape[N,H*W,C]
//          Can become: [N,C,H,W] → Reshape[N,C,H*W] → Transpose[0,2,1]
struct MoveTransposeThroughReshape : public OpRewritePattern<ONNXReshapeOp> {
  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReshapeOp reshapeOp, PatternRewriter &rewriter) const override {
    // Match pattern: Transpose → Reshape
    auto transposeOp = reshapeOp.getData().getDefiningOp<ONNXTransposeOp>();
    if (!transposeOp)
      return failure();

    auto perm = getTransposePermutation(transposeOp);
    if (!perm)
      return failure();

    auto transposeInputType =
        mlir::dyn_cast<RankedTensorType>(transposeOp.getData().getType());
    auto transposeOutputType =
        mlir::dyn_cast<RankedTensorType>(transposeOp.getType());
    auto reshapeOutputType =
        mlir::dyn_cast<RankedTensorType>(reshapeOp.getType());

    if (!transposeInputType || !transposeOutputType || !reshapeOutputType)
      return failure();

    if (isa<quant::UniformQuantizedPerAxisType>(
            transposeInputType.getElementType()))
      return failure();

    auto transposeOutputShape = transposeOutputType.getShape();
    auto reshapeOutputShape = reshapeOutputType.getShape();

    // Check if reshape is "factorizable" - only splits/merges within
    // each transposed dimension (no cross-dimension data mixing)
    SmallVector<SmallVector<int64_t>> dimGroups;
    if (!isSafeToSwapTransposeReshape(
            transposeOutputShape, reshapeOutputShape, dimGroups, *perm))
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "Fusing Transpose → Reshape into Reshape → Transpose\n");

    // Compute new reshape shape (to apply before transpose)
    SmallVector<int64_t> newReshapeShape =
        computePreTransposeReshapeShape(dimGroups, *perm);

    // Compute new transpose permutation (to apply after new reshape)
    SmallVector<int64_t> newPerm = computeAdjustedPermutation(dimGroups, *perm);

    // Create new Reshape (before transpose)
    auto newReshapeShapeType = RankedTensorType::get(
        {static_cast<int64_t>(newReshapeShape.size())}, rewriter.getI64Type());
    auto newReshapeShapeAttr = DenseElementsAttr::get(
        newReshapeShapeType, ArrayRef<int64_t>(newReshapeShape));
    auto newShapeConst = rewriter.create<ONNXConstantOp>(reshapeOp.getLoc(),
        newReshapeShapeType, Attribute(), newReshapeShapeAttr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);

    auto newReshapeType = RankedTensorType::get(
        newReshapeShape, transposeInputType.getElementType());
    auto newReshape = rewriter.create<ONNXReshapeOp>(transposeOp.getLoc(),
        newReshapeType, transposeOp.getData(), newShapeConst.getResult(),
        reshapeOp.getAllowzeroAttr());

    // Create new Transpose (after reshape)
    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(reshapeOp, reshapeOp.getType(),
        newReshape.getResult(), rewriter.getI64ArrayAttr(newPerm));

    return success();
  }

private:
  // Factorization result: maps each reshape dim to which transpose dim(s) it
  // came from
  struct DimFactorization {
    SmallVector<size_t> transposeIndices; // Which transpose dims contribute
    SmallVector<int64_t> reshapeSizes;    // Sizes after factorization
  };

  // Check if reshape only splits/merges within each dimension
  // Returns true if safe, and populates dimGroups with the factorization
  // perm: the transpose permutation (perm[output_dim] = input_dim)
  static bool isSafeToSwapTransposeReshape(
      ArrayRef<int64_t> transposeOutputShape,
      ArrayRef<int64_t> reshapeOutputShape,
      SmallVector<SmallVector<int64_t>> &outDimGroups, ArrayRef<int64_t> perm) {
    outDimGroups.clear();

    // Check for dynamic dimensions
    for (auto dim : transposeOutputShape) {
      if (dim <= 0)
        return false;
    }
    for (auto dim : reshapeOutputShape) {
      if (dim <= 0)
        return false;
    }

    // Calculate total element counts
    int64_t transposeTotal = 1;
    for (auto dim : transposeOutputShape)
      transposeTotal *= dim;
    int64_t reshapeTotal = 1;
    for (auto dim : reshapeOutputShape)
      reshapeTotal *= dim;

    // Must preserve total element count
    if (transposeTotal != reshapeTotal)
      return false;

    // Perform bidirectional factorization: handle both splitting (1→N) and
    // merging (N→1) We try to match reshape dims with transposed dims
    size_t transposeIdx = 0;
    size_t reshapeIdx = 0;

    while (transposeIdx < transposeOutputShape.size() &&
           reshapeIdx < reshapeOutputShape.size()) {
      int64_t transposeSize = transposeOutputShape[transposeIdx];
      int64_t reshapeSize = reshapeOutputShape[reshapeIdx];

      if (transposeSize == reshapeSize) {
        // Case 1: Exact match (identity) - one transpose dim → one reshape dim
        outDimGroups.push_back({reshapeSize});
        transposeIdx++;
        reshapeIdx++;
      } else if (transposeSize > reshapeSize) {
        // Case 2: Splitting - one transpose dim → multiple reshape dims
        SmallVector<int64_t> group;
        int64_t accumulatedSize = 1;

        // Collect consecutive reshape dims that multiply to this transpose dim
        while (reshapeIdx < reshapeOutputShape.size() &&
               accumulatedSize < transposeSize) {
          int64_t nextDim = reshapeOutputShape[reshapeIdx];
          int64_t newAccumulated = accumulatedSize * nextDim;

          // Stop if we would overshoot
          if (newAccumulated > transposeSize)
            return false; // Can't factor cleanly

          group.push_back(nextDim);
          accumulatedSize = newAccumulated;
          reshapeIdx++;
        }

        if (accumulatedSize != transposeSize)
          return false; // Didn't match exactly

        outDimGroups.push_back(group);
        transposeIdx++;
      } else {
        // Case 3: Merging - multiple transpose dims → one reshape dim
        // Accumulate consecutive transpose dims until we match the reshape dim
        int64_t accumulatedSize = transposeSize;
        size_t startIdx = transposeIdx;
        transposeIdx++;

        while (transposeIdx < transposeOutputShape.size() &&
               accumulatedSize < reshapeSize) {
          accumulatedSize *= transposeOutputShape[transposeIdx];
          transposeIdx++;
        }

        if (accumulatedSize != reshapeSize)
          return false; // Can't merge cleanly

        // FIX: When merging dimensions, we need to ensure that the merged
        // transpose output dimensions correspond to consecutive AND ascending
        // input dimensions. This is because reshape merges in memory order,
        // and if the input dims are not consecutive/ascending, the data will
        // be incorrectly reordered.
        //
        // For merge to be safe, the input dims (perm[i] for each merged output
        // dim i) must be consecutive (e.g., [2,3,4]) AND in ascending order.
        if (transposeIdx - startIdx > 1) {
          // Collect input dims for the merged output dims
          SmallVector<int64_t> inputDims;
          for (size_t i = startIdx; i < transposeIdx; ++i) {
            inputDims.push_back(perm[i]);
          }

          // Check 1: Input dims must be in ascending order in the perm
          // (meaning they appear in memory order in the output)
          for (size_t i = 1; i < inputDims.size(); ++i) {
            if (inputDims[i] <= inputDims[i - 1]) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Rejecting merge: input dims not ascending (["
                         << inputDims[i - 1] << "," << inputDims[i] << "])\n");
              return false;
            }
          }

          // Check 2: Input dims must be consecutive (no gaps)
          for (size_t i = 1; i < inputDims.size(); ++i) {
            if (inputDims[i] != inputDims[i - 1] + 1) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Rejecting merge: input dims not consecutive (["
                         << inputDims[i - 1] << "," << inputDims[i] << "])\n");
              return false;
            }
          }

          LLVM_DEBUG(llvm::dbgs() << "Merge of " << (transposeIdx - startIdx)
                                  << " dims is safe (consecutive ascending)\n");
        }

        // Record merge: multiple transpose dims map to single reshape dim
        // We use negative values to indicate merged dimensions
        // The first group gets the reshape size, subsequent groups get -1
        outDimGroups.push_back({reshapeSize});
        for (size_t i = startIdx + 1; i < transposeIdx; ++i) {
          outDimGroups.push_back({-1}); // Marker for merged dimension
        }

        reshapeIdx++;
      }
    }

    // All dims must be consumed
    return transposeIdx == transposeOutputShape.size() &&
           reshapeIdx == reshapeOutputShape.size();
  }

  // Compute the reshape shape to apply before transpose
  //
  // dimGroups[i] = how transposed dimension i is factored in the reshape
  //                positive values = actual dims, -1 = merged (skip)
  // perm[i] = which original dimension goes to transposed position i
  // invPerm[i] = which transposed position does original dimension i go to
  // We want: new shape in original dimension order
  [[nodiscard]] static SmallVector<int64_t> computePreTransposeReshapeShape(
      const SmallVector<SmallVector<int64_t>> &dimGroups,
      ArrayRef<int64_t> perm) {
    SmallVector<int64_t> newShape;
    auto invPerm = inversePermutation(perm);

    // For merging: we need to check if consecutive original dims are merged
    // and combine them BEFORE transpose
    SmallVector<bool> processed(perm.size(), false);

    for (size_t origDim = 0; origDim < perm.size(); ++origDim) {
      if (processed[origDim])
        continue;

      size_t transposedPos = invPerm[origDim];
      const auto &group = dimGroups[transposedPos];

      if (group.size() == 1 && group[0] == -1) {
        // This dimension was merged - skip it
        processed[origDim] = true;
        continue;
      }

      if (!group.empty() && group[0] > 0) {
        // Check if this starts a merge sequence
        // Look ahead to see if subsequent original dims are also merged
        for (size_t checkDim = origDim; checkDim < perm.size(); ++checkDim) {
          size_t checkTransposedPos = invPerm[checkDim];
          const auto &checkGroup = dimGroups[checkTransposedPos];

          if (checkGroup.empty())
            break;

          // If we hit a -1, this dim is part of a merge
          if (checkGroup.size() == 1 && checkGroup[0] == -1) {
            processed[checkDim] = true;
          } else if (checkDim == origDim) {
            // First dim in potential merge - continue
          } else {
            // Not part of this merge sequence
            break;
          }
        }

        // Append the group for this dimension (or merged dimensions)
        newShape.append(group.begin(), group.end());
        processed[origDim] = true;
      }
    }

    return newShape;
  }

  // Compute the adjusted transpose permutation for the new shape
  //
  // After applying the new reshape, we have a tensor where each original
  // dimension may be split into multiple sub-dimensions. We need to compute
  // the permutation that achieves the same final layout.
  [[nodiscard]] static SmallVector<int64_t> computeAdjustedPermutation(
      const SmallVector<SmallVector<int64_t>> &dimGroups,
      ArrayRef<int64_t> perm) {
    SmallVector<int64_t> newPerm;
    auto invPerm = inversePermutation(perm);

    // Build ranges: for each original dim, what indices does it occupy?
    // Handle merges: -1 means this dim was merged into a previous one
    SmallVector<std::pair<size_t, size_t>> origDimRanges(perm.size());
    size_t currentIdx = 0;

    for (size_t origDim = 0; origDim < perm.size(); ++origDim) {
      // invPerm[origDim] = which transposed position this original dim goes to
      size_t transposedPos = invPerm[origDim];

      const auto &group = dimGroups[transposedPos];
      if (!group.empty() && group[0] == -1) {
        // This dimension was merged - mark as no range
        origDimRanges[origDim] = {SIZE_MAX, SIZE_MAX};
      } else if (!group.empty()) {
        origDimRanges[origDim] = {currentIdx, currentIdx + group.size()};
        currentIdx += group.size();
      } else {
        origDimRanges[origDim] = {0, 0};
      }
    }

    // Apply original permutation: for each output position, emit the indices
    // of the corresponding original dimension (skipping merged dims)
    for (size_t origDim : perm) {
      auto range = origDimRanges[origDim];
      // Skip merged dimensions (marked with SIZE_MAX)
      if (range.first == SIZE_MAX)
        continue;
      for (size_t idx = range.first; idx < range.second; ++idx) {
        newPerm.push_back(static_cast<int64_t>(idx));
      }
    }

    return newPerm;
  }
};

//===----------------------------------------------------------------------===//
// Pattern 3: Push Transpose Through Unary SISO Operations (21+ ops)
// Rule: unary_op(transpose(x)) -> transpose(unary_op(x))
//===----------------------------------------------------------------------===//

template <typename UnaryOp>
struct PushTransposeThroughUnaryOp : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      UnaryOp op, PatternRewriter &rewriter) const override {
    auto transposeOp =
        op.getOperand().template getDefiningOp<ONNXTransposeOp>();
    if (!transposeOp)
      return failure();

    auto perm = getTransposePermutation(transposeOp);
    if (!perm)
      return failure();

    auto outputType = mlir::cast<RankedTensorType>(op.getType());

    auto newOutputShape =
        permuteShape(outputType.getShape(), inversePermutation(*perm));
    auto newOutputType =
        RankedTensorType::get(newOutputShape, outputType.getElementType());

    LLVM_DEBUG(llvm::dbgs() << "Pushing transpose through "
                            << op->getName().getStringRef() << "\n");

    auto newOp = rewriter.create<UnaryOp>(
        op.getLoc(), newOutputType, transposeOp.getOperand());

    for (auto namedAttr : op->getAttrs()) {
      if (namedAttr.getName() == "ResultNames")
        continue;
      newOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(
        op, op.getType(), newOp.getResult(), rewriter.getI64ArrayAttr(*perm));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 3b: Push Transpose Through Clip (has 3 operands: input, min, max)
//===----------------------------------------------------------------------===//

struct PushTransposeThroughClip : public OpRewritePattern<ONNXClipOp> {
  using OpRewritePattern<ONNXClipOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXClipOp op, PatternRewriter &rewriter) const override {
    // ONNXClipOp has input, min, max operands
    auto transposeOp = op.getOperand(0).getDefiningOp<ONNXTransposeOp>();
    if (!transposeOp)
      return failure();

    auto perm = getTransposePermutation(transposeOp);
    if (!perm)
      return failure();

    auto outputType = mlir::cast<RankedTensorType>(op.getType());

    auto newOutputShape =
        permuteShape(outputType.getShape(), inversePermutation(*perm));
    auto newOutputType =
        RankedTensorType::get(newOutputShape, outputType.getElementType());

    LLVM_DEBUG(llvm::dbgs() << "Pushing transpose through Clip\n");

    auto newOp = rewriter.create<ONNXClipOp>(op.getLoc(), newOutputType,
        transposeOp.getOperand(), op.getOperand(1), op.getOperand(2));

    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(
        op, op.getType(), newOp.getResult(), rewriter.getI64ArrayAttr(*perm));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 3c: Push Transpose Through HardSigmoid (has attributes, not extra
// operands)
//===----------------------------------------------------------------------===//

struct PushTransposeThroughHardSigmoid
    : public OpRewritePattern<ONNXHardSigmoidOp> {
  using OpRewritePattern<ONNXHardSigmoidOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXHardSigmoidOp op, PatternRewriter &rewriter) const override {
    // ONNXHardSigmoidOp has a single input operand
    auto transposeOp = op.getX().getDefiningOp<ONNXTransposeOp>();
    if (!transposeOp)
      return failure();

    auto perm = getTransposePermutation(transposeOp);
    if (!perm)
      return failure();

    auto outputType = mlir::cast<RankedTensorType>(op.getType());

    auto newOutputShape =
        permuteShape(outputType.getShape(), inversePermutation(*perm));
    auto newOutputType =
        RankedTensorType::get(newOutputShape, outputType.getElementType());

    LLVM_DEBUG(llvm::dbgs() << "Pushing transpose through HardSigmoid"
                            << "\n");

    auto newOp = rewriter.create<ONNXHardSigmoidOp>(op.getLoc(), newOutputType,
        transposeOp.getOperand(), op.getAlphaAttr(), op.getBetaAttr());

    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(
        op, op.getType(), newOp.getResult(), rewriter.getI64ArrayAttr(*perm));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 4: Push Transpose Through QuantizeLinear/DequantizeLinear (6 ops)
// Rule: qdq(transpose(x), scale, zp) -> transpose(qdq(x, scale, zp))
//===----------------------------------------------------------------------===//

template <typename QDQOp>
struct PushTransposeThroughQDQ : public OpRewritePattern<QDQOp> {
  using OpRewritePattern<QDQOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      QDQOp op, PatternRewriter &rewriter) const override {
    auto transposeOp =
        op->getOperand(0).template getDefiningOp<ONNXTransposeOp>();
    if (!transposeOp)
      return failure();

    auto perm = getTransposePermutation(transposeOp);
    if (!perm)
      return failure();

    auto outputType = mlir::cast<RankedTensorType>(op.getType());
    auto newOutputShape =
        permuteShape(outputType.getShape(), inversePermutation(*perm));
    auto newOutputType =
        RankedTensorType::get(newOutputShape, outputType.getElementType());

    LLVM_DEBUG(llvm::dbgs() << "Pushing transpose through QDQ "
                            << op->getName().getStringRef() << "\n");

    SmallVector<Value> operands;
    operands.push_back(transposeOp.getOperand()); // Transpose is unary, no (0)
    for (size_t i = 1; i < op->getNumOperands(); ++i) {
      operands.push_back(op->getOperand(i));
    }

    // FIX: Transform axis attribute for per-channel quantization
    // When pushing transpose through QDQ, the axis must be transformed
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : op->getAttrs()) {
      if (attr.getName() == "ResultNames")
        continue;
      if (attr.getName() == "axis") {
        if (auto axisAttr = mlir::dyn_cast<IntegerAttr>(attr.getValue())) {
          int64_t oldAxis = axisAttr.getValue().getSExtValue();
          int64_t rank = static_cast<int64_t>(perm->size());
          if (oldAxis < 0)
            oldAxis += rank;
          if (oldAxis >= 0 && oldAxis < rank) {
            // Transform axis: newAxis = perm[oldAxis]
            int64_t newAxis = (*perm)[oldAxis];
            // ONNX dialect expects si64 (signed 64-bit) for axis attribute
            auto si64Type = rewriter.getIntegerType(64, /*isSigned=*/true);
            newAttrs.push_back(rewriter.getNamedAttr(
                "axis", IntegerAttr::get(si64Type, newAxis)));
            LLVM_DEBUG(llvm::dbgs() << "Transformed QDQ axis: " << oldAxis
                                    << " -> " << newAxis << "\n");
            continue;
          }
        }
      }
      newAttrs.push_back(attr);
    }

    auto newOp =
        rewriter.create<QDQOp>(op.getLoc(), newOutputType, operands, newAttrs);

    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(
        op, op.getType(), newOp.getResult(), rewriter.getI64ArrayAttr(*perm));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 4b: Push Transpose Through quant.scast
// Rule: scast(transpose(x)) -> transpose(scast(x))
// quant.scast is a pure type-cast (quantized <-> storage) that doesn't
// change data layout, so transpose can be pushed through it.
//===----------------------------------------------------------------------===//

struct PushTransposeThroughSCast
    : public OpRewritePattern<quant::StorageCastOp> {
  using OpRewritePattern<quant::StorageCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      quant::StorageCastOp op, PatternRewriter &rewriter) const override {
    auto transposeOp = op.getOperand().getDefiningOp<ONNXTransposeOp>();
    if (!transposeOp)
      return failure();

    auto perm = getTransposePermutation(transposeOp);
    if (!perm)
      return failure();

    if (llvm::any_of(op->getUsers(), [](Operation *op) {
          return isa<ONNXDequantizeLinearOp, func::ReturnOp>(op);
        }))
      return rewriter.notifyMatchFailure(
          op, "Not pushing through boundary scast");

    auto outputType = mlir::cast<RankedTensorType>(op.getType());

    // The new scast takes the transpose's input directly, so its output must
    // have the same shape as that input (scast only changes the element type).
    // For per-axis quant types, remap the quant axis from post-transpose
    // (output) space to pre-transpose (input) space.
    auto inputType =
        mlir::cast<RankedTensorType>(transposeOp.getOperand().getType());
    Type newElemType = outputType.getElementType();
    if (auto perAxisType =
            dyn_cast<quant::UniformQuantizedPerAxisType>(newElemType)) {
      int32_t oldAxis = perAxisType.getQuantizedDimension();
      if (oldAxis >= 0 && oldAxis < static_cast<int32_t>(perm->size())) {
        int32_t newAxis = static_cast<int32_t>((*perm)[oldAxis]);
        if (newAxis != oldAxis) {
          newElemType = quant::UniformQuantizedPerAxisType::get(
              perAxisType.getFlags(), perAxisType.getStorageType(),
              perAxisType.getExpressedType(), perAxisType.getScales(),
              perAxisType.getZeroPoints(), newAxis,
              perAxisType.getStorageTypeMin(), perAxisType.getStorageTypeMax());
        }
      }
    }
    auto newOutputType =
        RankedTensorType::get(inputType.getShape(), newElemType);

    LLVM_DEBUG(llvm::dbgs() << "Pushing transpose through quant.scast\n");

    // Create scast before transpose: scast(transpose_input)
    auto newSCast = rewriter.create<quant::StorageCastOp>(
        op.getLoc(), newOutputType, transposeOp.getOperand());

    // Create transpose after scast
    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(op, op.getType(),
        newSCast.getResult(), rewriter.getI64ArrayAttr(*perm));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 5: Fuse Binary Operations with Both Inputs Transposed (8 ops)
// Rule: binop(transpose(x, p), transpose(y, p)) -> transpose(binop(x, y), p)
//===----------------------------------------------------------------------===//

template <typename BinaryOp>
struct FuseBinaryOpTransposes : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      BinaryOp op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2)
      return failure();

    auto lhsTranspose =
        op->getOperand(0).template getDefiningOp<ONNXTransposeOp>();
    auto rhsTranspose =
        op->getOperand(1).template getDefiningOp<ONNXTransposeOp>();

    if (!lhsTranspose || !rhsTranspose)
      return failure();

    auto lhsPerm = getTransposePermutation(lhsTranspose);
    auto rhsPerm = getTransposePermutation(rhsTranspose);

    if (!lhsPerm || !rhsPerm || *lhsPerm != *rhsPerm)
      return failure();

    auto outputType = mlir::cast<RankedTensorType>(op.getType());
    auto newOutputShape =
        permuteShape(outputType.getShape(), inversePermutation(*lhsPerm));
    auto newOutputType =
        RankedTensorType::get(newOutputShape, outputType.getElementType());

    LLVM_DEBUG(llvm::dbgs() << "Fusing transposes through binary op "
                            << op->getName().getStringRef() << "\n");

    auto newOp = rewriter.create<BinaryOp>(op.getLoc(), newOutputType,
        lhsTranspose.getOperand(), rhsTranspose.getOperand());

    for (auto namedAttr : op->getAttrs()) {
      if (namedAttr.getName() == "ResultNames")
        continue;
      newOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(op, op.getType(),
        newOp.getResult(), rewriter.getI64ArrayAttr(*lhsPerm));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 6: Binary Op with One Transpose and Transpose-Immune Operand (6 ops)
// Rule: binop(transpose(x), y) -> transpose(binop(x, reshape(y)))
//       where y is transpose-immune (scalar or single non-1 dim)
//===----------------------------------------------------------------------===//

template <typename BinaryOp>
struct FuseTransposeImmuneBinaryOp : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      BinaryOp op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2)
      return failure();

    auto lhsTranspose =
        op->getOperand(0).template getDefiningOp<ONNXTransposeOp>();
    Value otherOperand = op->getOperand(1);
    bool isLhsTransposed = true;

    if (!lhsTranspose) {
      lhsTranspose =
          op->getOperand(1).template getDefiningOp<ONNXTransposeOp>();
      otherOperand = op->getOperand(0);
      isLhsTransposed = false;
    }

    if (!lhsTranspose)
      return failure();

    auto otherType = mlir::dyn_cast<RankedTensorType>(otherOperand.getType());
    if (!otherType || !isTransposeImmune(otherType.getShape()))
      return failure();

    auto perm = getTransposePermutation(lhsTranspose);
    if (!perm)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Fusing transpose-immune binary op "
                            << op->getName().getStringRef() << "\n");

    auto invPerm = inversePermutation(*perm);

    // Handle rank mismatch by expanding lower-rank operand with 1s
    SmallVector<int64_t> expandedShape;
    if (invPerm.size() > otherType.getShape().size()) {
      // Expand lower-rank shape by prepending 1s (e.g., [1] -> [1,1,1,1])
      expandedShape = expandShape(otherType.getShape(), invPerm.size());
      LLVM_DEBUG(llvm::dbgs() << "  Expanding shape from rank "
                              << otherType.getShape().size() << " to rank "
                              << invPerm.size() << "\n");
    } else if (invPerm.size() < otherType.getShape().size()) {
      // Higher-rank operand - cannot safely expand transpose
      LLVM_DEBUG(llvm::dbgs()
                 << "  Permutation rank (" << invPerm.size()
                 << ") < shape rank (" << otherType.getShape().size()
                 << "), skipping fusion\n");
      return failure();
    } else {
      // Same rank - use as-is
      expandedShape = SmallVector<int64_t>(otherType.getShape());
    }

    auto newShape = permuteShape(expandedShape, invPerm);

    // Check if reshape is actually needed (shape might be unchanged for
    // 1x1x1x1)
    Value reshapedOperand;
    if (newShape == otherType.getShape()) {
      // Shape unchanged - use operand directly (no-op Reshape avoided)
      reshapedOperand = otherOperand;
      LLVM_DEBUG(llvm::dbgs() << "  Shape unchanged, skipping Reshape\n");
    } else {
      // Shape changed - need Reshape.
      // Remap per-axis quant dimension through the inverse permutation.
      auto reshapedElemType =
          remapPerAxisQuantType(otherType.getElementType(), invPerm);
      auto newType = RankedTensorType::get(newShape, reshapedElemType);

      // Create a constant for the new shape
      auto shapeType = RankedTensorType::get(
          {static_cast<int64_t>(newShape.size())}, rewriter.getI64Type());
      auto shapeAttr =
          DenseElementsAttr::get(shapeType, ArrayRef<int64_t>(newShape));
      auto shapeConst =
          rewriter.create<ONNXConstantOp>(op.getLoc(), shapeType, Attribute(),
              shapeAttr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

      auto allowzeroAttr =
          rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), 0);
      auto reshapeOp =
          rewriter.create<ONNXReshapeOp>(op.getLoc(), newType, otherOperand,
              /*shape=*/shapeConst.getResult(), /*allowzero=*/allowzeroAttr);

      reshapedOperand = reshapeOp.getResult();
    }

    auto outputType = mlir::cast<RankedTensorType>(op.getType());
    auto newOutputShape =
        permuteShape(outputType.getShape(), inversePermutation(*perm));
    auto newOutputType =
        RankedTensorType::get(newOutputShape, outputType.getElementType());

    Value lhs = isLhsTransposed ? lhsTranspose.getOperand() : reshapedOperand;
    Value rhs = isLhsTransposed ? reshapedOperand : lhsTranspose.getOperand();

    auto newOp =
        rewriter.create<BinaryOp>(op.getLoc(), newOutputType, lhs, rhs);

    for (auto namedAttr : op->getAttrs()) {
      if (namedAttr.getName() == "ResultNames")
        continue;
      newOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(
        op, op.getType(), newOp.getResult(), rewriter.getI64ArrayAttr(*perm));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 6b: Binary Op with One Transpose and Constant Operand
// Rule: binop(transpose(x), const) -> transpose(binop(x, transposed_const))
//       where const is NOT transpose-immune (needs data transposition)
//===----------------------------------------------------------------------===//

template <typename BinaryOp>
struct PushTransposeThroughBinaryWithConst : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      BinaryOp op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2)
      return failure();

    // Try to find transpose + constant pattern
    Value transposeValue;
    Value constantValue;
    ONNXTransposeOp transposeOp;
    ONNXConstantOp constantOp;
    bool isLhsTransposed = false;

    // Check LHS transpose, RHS constant
    auto lhsTranspose =
        op->getOperand(0).template getDefiningOp<ONNXTransposeOp>();
    auto rhsConstant =
        op->getOperand(1).template getDefiningOp<ONNXConstantOp>();

    if (lhsTranspose && rhsConstant) {
      transposeOp = lhsTranspose;
      constantOp = rhsConstant;
      transposeValue = op->getOperand(0);
      constantValue = op->getOperand(1);
      isLhsTransposed = true;
    } else {
      // Check RHS transpose, LHS constant
      auto rhsTranspose =
          op->getOperand(1).template getDefiningOp<ONNXTransposeOp>();
      auto lhsConstant =
          op->getOperand(0).template getDefiningOp<ONNXConstantOp>();

      if (rhsTranspose && lhsConstant) {
        transposeOp = rhsTranspose;
        constantOp = lhsConstant;
        transposeValue = op->getOperand(1);
        constantValue = op->getOperand(0);
        isLhsTransposed = false;
      } else {
        return failure();
      }
    }

    auto perm = getTransposePermutation(transposeOp);
    if (!perm)
      return failure();

    auto constType = mlir::dyn_cast<RankedTensorType>(constantValue.getType());
    if (!constType)
      return failure();

    // Skip if constant is transpose-immune (handled by different pattern)
    if (isTransposeImmune(constType.getShape()))
      return failure();

    // Handle rank mismatch: expand constant shape by prepending 1s
    // (matches ONNX right-aligned broadcast semantics).
    SmallVector<int64_t> constShape(
        constType.getShape().begin(), constType.getShape().end());
    if (constShape.size() < perm->size()) {
      size_t diff = perm->size() - constShape.size();
      SmallVector<int64_t> expanded(diff, 1);
      expanded.append(constShape.begin(), constShape.end());
      constShape = expanded;
    } else if (constShape.size() > perm->size()) {
      return failure();
    }

    // Get the constant's value attribute
    auto valueAttr = constantOp.getValueAttr();
    if (!valueAttr)
      return failure();

    auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(valueAttr);
    if (!denseAttr)
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "Pushing transpose through binary op with constant "
               << op->getName().getStringRef() << "\n");

    // Transpose the constant's data (using possibly expanded shape)
    auto invPerm = inversePermutation(*perm);
    auto newConstShape = permuteShape(constShape, invPerm);

    // Remap per-axis quant dimension through the inverse transpose applied
    // to the constant (data moves from post-transpose to pre-transpose space).
    auto origElementType = constType.getElementType();
    auto remappedElementType = remapPerAxisQuantType(origElementType, invPerm);
    auto isQuantized = mlir::isa<mlir::quant::QuantizedType>(origElementType);

    // For quantized types, we need to work with storage type for
    // DenseElementsAttr
    auto workingElementType = origElementType;
    if (isQuantized) {
      auto quantType = mlir::cast<mlir::quant::QuantizedType>(origElementType);
      workingElementType = quantType.getStorageType();
    }

    // Transpose the dense data
    SmallVector<int64_t> strides(constShape.size());
    int64_t stride = 1;
    for (int i = constShape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= constShape[i];
    }

    // Calculate new strides after transpose
    SmallVector<int64_t> newStrides(newConstShape.size());
    stride = 1;
    for (int i = newConstShape.size() - 1; i >= 0; --i) {
      newStrides[i] = stride;
      stride *= newConstShape[i];
    }

    // Create mapping from new indices to old indices
    int64_t numElements = 1;
    for (auto dim : constShape)
      numElements *= dim;

    SmallVector<Attribute> newValues;
    newValues.reserve(numElements);

    // For each position in the new transposed tensor, compute the old position
    auto rawData = denseAttr.getRawData();

    // Splat constants have compressed storage - transpose directly
    if (denseAttr.isSplat()) {
      RankedTensorType denseAttrType;
      if (isQuantized) {
        denseAttrType =
            RankedTensorType::get(newConstShape, workingElementType);
      } else {
        denseAttrType = RankedTensorType::get(newConstShape, origElementType);
      }
      auto newDenseAttr = DenseElementsAttr::get(
          denseAttrType, denseAttr.getSplatValue<Attribute>());
      auto newConstType =
          RankedTensorType::get(newConstShape, remappedElementType);
      auto newConstOp = rewriter.create<ONNXConstantOp>(constantOp.getLoc(),
          newConstType, Attribute(), newDenseAttr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr);

      auto outputType = mlir::cast<RankedTensorType>(op.getType());
      auto newOutputShape = permuteShape(outputType.getShape(), invPerm);
      auto newOutputType =
          RankedTensorType::get(newOutputShape, outputType.getElementType());

      Value lhs =
          isLhsTransposed ? transposeOp.getOperand() : newConstOp.getResult();
      Value rhs =
          isLhsTransposed ? newConstOp.getResult() : transposeOp.getOperand();

      auto newOp =
          rewriter.create<BinaryOp>(op.getLoc(), newOutputType, lhs, rhs);

      for (auto namedAttr : op->getAttrs()) {
        if (namedAttr.getName() == "ResultNames")
          continue;
        newOp->setAttr(namedAttr.getName(), namedAttr.getValue());
      }

      rewriter.replaceOpWithNewOp<ONNXTransposeOp>(
          op, op.getType(), newOp.getResult(), rewriter.getI64ArrayAttr(*perm));

      return success();
    }

    if (workingElementType.isF32()) {
      // Handle float32
      // Use ArrayRef for safe access without pointer arithmetic
      ArrayRef<float> floatData(reinterpret_cast<const float *>(rawData.data()),
          static_cast<size_t>(numElements));
      for (int64_t newLinearIdx = 0; newLinearIdx < numElements;
          ++newLinearIdx) {
        // Convert linear index to multi-dimensional index in new space
        SmallVector<int64_t> newIndices(newConstShape.size());
        int64_t remaining = newLinearIdx;
        for (int i = newConstShape.size() - 1; i >= 0; --i) {
          newIndices[i] = remaining % newConstShape[i];
          remaining /= newConstShape[i];
        }

        // Map to old indices using inverse permutation
        SmallVector<int64_t> oldIndices(constShape.size());
        for (size_t i = 0; i < invPerm.size(); ++i) {
          oldIndices[invPerm[i]] = newIndices[i];
        }

        // Convert old multi-dimensional index to linear index
        int64_t oldLinearIdx = 0;
        for (size_t i = 0; i < oldIndices.size(); ++i) {
          oldLinearIdx += oldIndices[i] * strides[i];
        }

        // Copy the element using safe ArrayRef access
        float value = floatData[static_cast<size_t>(oldLinearIdx)];
        newValues.push_back(rewriter.getF32FloatAttr(value));
      }
    } else if (workingElementType.isInteger(64)) {
      // Handle int64
      // Use ArrayRef for safe access without pointer arithmetic
      ArrayRef<int64_t> intData(
          reinterpret_cast<const int64_t *>(rawData.data()),
          static_cast<size_t>(numElements));
      for (int64_t newLinearIdx = 0; newLinearIdx < numElements;
          ++newLinearIdx) {
        SmallVector<int64_t> newIndices(newConstShape.size());
        int64_t remaining = newLinearIdx;
        for (int i = newConstShape.size() - 1; i >= 0; --i) {
          newIndices[i] = remaining % newConstShape[i];
          remaining /= newConstShape[i];
        }

        SmallVector<int64_t> oldIndices(constShape.size());
        for (size_t i = 0; i < invPerm.size(); ++i) {
          oldIndices[invPerm[i]] = newIndices[i];
        }

        int64_t oldLinearIdx = 0;
        for (size_t i = 0; i < oldIndices.size(); ++i) {
          oldLinearIdx += oldIndices[i] * strides[i];
        }

        // Copy the element using safe ArrayRef access
        int64_t value = intData[static_cast<size_t>(oldLinearIdx)];
        newValues.push_back(rewriter.getI64IntegerAttr(value));
      }
    } else if (workingElementType.isInteger(8)) {
      // Handle int8/uint8 (quantized types)
      // Use ArrayRef for safe access without pointer arithmetic
      ArrayRef<int8_t> intData(reinterpret_cast<const int8_t *>(rawData.data()),
          static_cast<size_t>(numElements));
      for (int64_t newLinearIdx = 0; newLinearIdx < numElements;
          ++newLinearIdx) {
        // Convert linear index to multi-dimensional index in new space
        SmallVector<int64_t> newIndices(newConstShape.size());
        int64_t remaining = newLinearIdx;
        for (int i = newConstShape.size() - 1; i >= 0; --i) {
          newIndices[i] = remaining % newConstShape[i];
          remaining /= newConstShape[i];
        }

        // Map to old indices using inverse permutation
        SmallVector<int64_t> oldIndices(constShape.size());
        for (size_t i = 0; i < invPerm.size(); ++i) {
          oldIndices[invPerm[i]] = newIndices[i];
        }

        // Convert old multi-dimensional index to linear index
        int64_t oldLinearIdx = 0;
        for (size_t i = 0; i < oldIndices.size(); ++i) {
          oldLinearIdx += oldIndices[i] * strides[i];
        }

        // Copy the element using safe ArrayRef access
        int8_t value = intData[static_cast<size_t>(oldLinearIdx)];
        // Reinterpret as unsigned bits to avoid APInt assertion for signed
        // values
        uint64_t rawValue = static_cast<uint8_t>(value);
        newValues.push_back(
            rewriter.getIntegerAttr(workingElementType, APInt(8, rawValue)));
      }
    } else {
      // Unsupported type - skip this optimization
      return failure();
    }

    // Create new constant with transposed data
    // For quantized types, create DenseElementsAttr with storage type
    RankedTensorType denseAttrType;
    if (isQuantized) {
      denseAttrType = RankedTensorType::get(newConstShape, workingElementType);
    } else {
      denseAttrType = RankedTensorType::get(newConstShape, origElementType);
    }

    auto newDenseAttr = DenseElementsAttr::get(denseAttrType, newValues);

    // Create the constant with the final type (including quantization),
    // using the remapped per-axis quant dimension.
    auto newConstType =
        RankedTensorType::get(newConstShape, remappedElementType);
    auto newConstOp = rewriter.create<ONNXConstantOp>(constantOp.getLoc(),
        newConstType, Attribute(), newDenseAttr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr);

    // Create new binary op with untransposed input and transposed constant
    auto outputType = mlir::cast<RankedTensorType>(op.getType());
    auto newOutputShape = permuteShape(outputType.getShape(), invPerm);
    auto newOutputType =
        RankedTensorType::get(newOutputShape, outputType.getElementType());

    Value lhs =
        isLhsTransposed ? transposeOp.getOperand() : newConstOp.getResult();
    Value rhs =
        isLhsTransposed ? newConstOp.getResult() : transposeOp.getOperand();

    auto newOp =
        rewriter.create<BinaryOp>(op.getLoc(), newOutputType, lhs, rhs);

    // Copy attributes
    for (auto namedAttr : op->getAttrs()) {
      if (namedAttr.getName() == "ResultNames")
        continue;
      newOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Add transpose after the binary op
    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(
        op, op.getType(), newOp.getResult(), rewriter.getI64ArrayAttr(*perm));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 6c: Fold Const + DequantizeLinear + Transpose (Per-Channel)
// Transposes per-channel quantized constants by transposing data and axis
//===----------------------------------------------------------------------===//

struct FoldConstDQTranspose : public OpRewritePattern<ONNXTransposeOp> {
  using OpRewritePattern<ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXTransposeOp transposeOp, PatternRewriter &rewriter) const override {
    // Match pattern: transpose(dequantize(const))
    auto dqOp = transposeOp.getData().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp)
      return failure();

    auto constOp = dqOp.getX().getDefiningOp<ONNXConstantOp>();
    if (!constOp)
      return failure();

    // Get transpose permutation
    auto permAttr = transposeOp.getPermAttr();
    if (!permAttr)
      return failure();
    SmallVector<int64_t> perm;
    for (auto attr : permAttr.getValue())
      perm.push_back(mlir::cast<IntegerAttr>(attr).getInt());

    // Get constant's value attribute
    auto valueAttr = constOp.getValueAttr();
    if (!valueAttr)
      return failure();

    auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(valueAttr);
    if (!denseAttr)
      return failure();

    auto constType = mlir::cast<RankedTensorType>(constOp.getType());
    auto constShape = constType.getShape();

    // Validate permutation is valid
    if (perm.size() != constShape.size()) {
      return failure();
    }

    // Validate that perm is actually a valid permutation (contains 0..n-1)
    SmallVector<bool> seen(perm.size(), false);
    for (int64_t val : perm) {
      if (val < 0 || static_cast<size_t>(val) >= perm.size() || seen[val]) {
        return failure(); // Invalid permutation
      }
      seen[val] = true;
    }

    // Only handle per-channel quantization
    if (!dqOp.getAxisAttr())
      return failure();

    int64_t oldAxis = dqOp.getAxisAttr().getValue().getSExtValue();
    auto rank = static_cast<int64_t>(constShape.size());

    if (oldAxis < 0) {
      oldAxis += rank;
    }

    // Validate normalized axis is in bounds
    if (oldAxis < 0 || oldAxis >= rank)
      return failure();

    // Find the new position of the quantization axis after transposition
    // For transpose with perm, output[i] = input[perm[i]]
    // So if a dimension is at position oldAxis in input, we need to find
    // the position i in output where perm[i] == oldAxis
    int64_t newAxis = -1;
    for (size_t i = 0; i < perm.size(); ++i) {
      if (perm[i] == oldAxis) {
        newAxis = static_cast<int64_t>(i);
        break;
      }
    }

    if (newAxis < 0) {
      // This should never happen with a valid permutation
      return failure();
    }

    auto newConstShape = permuteShape(constShape, perm);

    if (static_cast<size_t>(newAxis) >= newConstShape.size())
      return failure();

    auto origElementType = constType.getElementType();
    auto isQuantized = mlir::isa<mlir::quant::QuantizedType>(origElementType);

    auto workingElementType = origElementType;
    if (isQuantized) {
      auto quantType = mlir::cast<mlir::quant::QuantizedType>(origElementType);
      workingElementType = quantType.getStorageType();
    }

    SmallVector<int64_t> strides(constShape.size());
    int64_t stride = 1;
    for (int i = constShape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= constShape[i];
    }

    int64_t numElements = std::accumulate(
        constShape.begin(), constShape.end(), 1LL, std::multiplies<int64_t>());

    SmallVector<Attribute> newValues;
    newValues.reserve(numElements);

    auto rawData = denseAttr.getRawData();

    // Validate raw data size matches expected tensor size
    size_t expectedBytes =
        numElements * (workingElementType.getIntOrFloatBitWidth() / 8);
    if (rawData.size() != expectedBytes) {
      // Data size mismatch - likely a splat/broadcast constant, skip
      // optimization
      return failure();
    }

    if (workingElementType.isF32()) {
      ArrayRef<float> floatData(reinterpret_cast<const float *>(rawData.data()),
          static_cast<size_t>(numElements));
      for (int64_t newLinearIdx = 0; newLinearIdx < numElements;
          ++newLinearIdx) {
        SmallVector<int64_t> newIndices(newConstShape.size());
        int64_t remaining = newLinearIdx;
        for (int i = newConstShape.size() - 1; i >= 0; --i) {
          newIndices[i] = remaining % newConstShape[i];
          remaining /= newConstShape[i];
        }

        SmallVector<int64_t> oldIndices(constShape.size());
        for (size_t i = 0; i < perm.size(); ++i) {
          oldIndices[perm[i]] = newIndices[i];
        }

        int64_t oldLinearIdx = 0;
        for (size_t i = 0; i < oldIndices.size(); ++i) {
          oldLinearIdx += oldIndices[i] * strides[i];
        }

        float value = floatData[static_cast<size_t>(oldLinearIdx)];
        newValues.push_back(rewriter.getF32FloatAttr(value));
      }
    } else if (workingElementType.isInteger(8) ||
               workingElementType.isInteger(16) ||
               workingElementType.isInteger(32) ||
               workingElementType.isInteger(64)) {
      unsigned bitWidth = workingElementType.getIntOrFloatBitWidth();
      ArrayRef<int8_t> intData(
          reinterpret_cast<const int8_t *>(rawData.data()), rawData.size());

      for (int64_t newLinearIdx = 0; newLinearIdx < numElements;
          ++newLinearIdx) {
        SmallVector<int64_t> newIndices(newConstShape.size());
        int64_t remaining = newLinearIdx;
        for (int i = newConstShape.size() - 1; i >= 0; --i) {
          newIndices[i] = remaining % newConstShape[i];
          remaining /= newConstShape[i];
        }

        SmallVector<int64_t> oldIndices(constShape.size());
        for (size_t i = 0; i < perm.size(); ++i) {
          oldIndices[perm[i]] = newIndices[i];
        }

        int64_t oldLinearIdx = 0;
        for (size_t i = 0; i < oldIndices.size(); ++i) {
          oldLinearIdx += oldIndices[i] * strides[i];
        }

        uint64_t rawValue = 0;
        auto byteOffset = static_cast<size_t>(oldLinearIdx * bitWidth / 8);

        if (bitWidth == 8) {
          rawValue = static_cast<uint8_t>(intData[byteOffset]);
        } else if (bitWidth == 16) {
          rawValue = static_cast<uint16_t>(
              *reinterpret_cast<const int16_t *>(&intData[byteOffset]));
        } else if (bitWidth == 32) {
          rawValue = static_cast<uint32_t>(
              *reinterpret_cast<const int32_t *>(&intData[byteOffset]));
        } else if (bitWidth == 64) {
          rawValue = static_cast<uint64_t>(
              *reinterpret_cast<const int64_t *>(&intData[byteOffset]));
        }

        newValues.push_back(rewriter.getIntegerAttr(
            workingElementType, APInt(bitWidth, rawValue)));
      }
    } else {
      return failure();
    }

    RankedTensorType denseAttrType;
    if (isQuantized) {
      denseAttrType = RankedTensorType::get(newConstShape, workingElementType);
    } else {
      denseAttrType = RankedTensorType::get(newConstShape, origElementType);
    }

    auto newDenseAttr = DenseElementsAttr::get(denseAttrType, newValues);
    auto newConstType = RankedTensorType::get(newConstShape, origElementType);
    auto newConstOp = rewriter.create<ONNXConstantOp>(constOp.getLoc(),
        newConstType, Attribute(), newDenseAttr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr);

    // Scale was already validated earlier
    Value newScale = dqOp.getXScale();
    Value newZeroPoint = dqOp.getXZeroPoint();

    auto transposeOutputType =
        mlir::cast<RankedTensorType>(transposeOp.getType());
    auto newDqOp = rewriter.create<ONNXDequantizeLinearOp>(dqOp.getLoc(),
        transposeOutputType, newConstOp.getResult(), newScale, newZeroPoint);

    for (auto namedAttr : dqOp->getAttrs()) {
      if (namedAttr.getName().strref() != "axis")
        newDqOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    auto axisAttr =
        IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
            APInt(64, static_cast<uint64_t>(newAxis)));
    newDqOp->setAttr(rewriter.getStringAttr("axis"), axisAttr);

    rewriter.replaceOp(transposeOp, newDqOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 7b: Push Transpose Through Where Operation
// Rule: where(cond, transpose(x), transpose(y)) -> transpose(where(cond, x, y))
//===----------------------------------------------------------------------===//

struct PushTransposeThroughWhere : public OpRewritePattern<ONNXWhereOp> {
  using OpRewritePattern<ONNXWhereOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXWhereOp op, PatternRewriter &rewriter) const override {
    auto xTranspose = op.getX().getDefiningOp<ONNXTransposeOp>();
    auto yTranspose = op.getY().getDefiningOp<ONNXTransposeOp>();

    if (!xTranspose || !yTranspose)
      return failure();

    auto xPerm = getTransposePermutation(xTranspose);
    auto yPerm = getTransposePermutation(yTranspose);

    if (!xPerm || !yPerm || *xPerm != *yPerm)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Pushing transpose through Where\n");

    auto outputType = mlir::cast<RankedTensorType>(op.getType());
    auto invPerm = inversePermutation(*xPerm);
    auto newOutputShape = permuteShape(outputType.getShape(), invPerm);
    auto newOutputType =
        RankedTensorType::get(newOutputShape, outputType.getElementType());

    // Check if condition also has transpose
    Value newCondition = op.getCondition();
    auto condTranspose = op.getCondition().getDefiningOp<ONNXTransposeOp>();
    if (condTranspose) {
      auto condPerm = getTransposePermutation(condTranspose);
      if (condPerm && *condPerm == *xPerm) {
        newCondition = condTranspose.getOperand();
      }
    }

    auto newWhere = rewriter.create<ONNXWhereOp>(op.getLoc(), newOutputType,
        newCondition, xTranspose.getOperand(), yTranspose.getOperand());

    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(op, op.getType(),
        newWhere.getResult(), rewriter.getI64ArrayAttr(*xPerm));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 7c: Push Transpose Through Variadic Min/Max with Constants
// Rule: min/max(transpose(x), const, ...) -> transpose(min/max(x,
// transposed_const, ...))
//===----------------------------------------------------------------------===//

template <typename VariadicOp>
struct PushTransposeThroughVariadicWithConst
    : public OpRewritePattern<VariadicOp> {
  using OpRewritePattern<VariadicOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      VariadicOp op, PatternRewriter &rewriter) const override {
    SmallVector<Value> transposeInputs;
    SmallVector<Value> constantInputs;
    SmallVector<Value> otherInputs;
    SmallVector<int64_t> firstPerm;
    ONNXTransposeOp firstTranspose;

    // Categorize inputs
    for (auto input : op->getOperands()) {
      if (auto transposeOp = input.template getDefiningOp<ONNXTransposeOp>()) {

        auto perm = getTransposePermutation(transposeOp);
        if (!perm)
          return failure();

        if (firstPerm.empty()) {
          firstPerm = *perm;
          firstTranspose = transposeOp;
        } else if (firstPerm != *perm) {
          return failure();
        }

        transposeInputs.push_back(transposeOp.getOperand());
      } else if (auto constOp =
                     input.template getDefiningOp<ONNXConstantOp>()) {
        constantInputs.push_back(input);
      } else {
        return failure();
      }
    }

    if (transposeInputs.empty() || firstPerm.empty())
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "Pushing transpose through variadic op with constant\n");

    // Transpose all constants
    SmallVector<Value> newInputs;
    newInputs.append(transposeInputs.begin(), transposeInputs.end());

    for (auto constValue : constantInputs) {
      auto constType = mlir::cast<RankedTensorType>(constValue.getType());

      // Handle rank mismatch: expand constant shape by prepending 1s
      SmallVector<int64_t> cShape(
          constType.getShape().begin(), constType.getShape().end());
      if (cShape.size() < firstPerm.size()) {
        size_t diff = firstPerm.size() - cShape.size();
        SmallVector<int64_t> expanded(diff, 1);
        expanded.append(cShape.begin(), cShape.end());
        cShape = expanded;
      } else if (cShape.size() > firstPerm.size()) {
        return failure();
      }

      // Check if transpose-immune
      if (isTransposeImmune(cShape)) {
        // For transpose-immune shapes (e.g. 1x1x1x1), permutation doesn't
        // change shape
        auto invPerm = inversePermutation(firstPerm);
        auto newShape = permuteShape(cShape, invPerm);

        // Only create Reshape if shape actually changes (it won't for 1x1x1x1)
        if (newShape == constType.getShape()) {
          // Shape unchanged - use constant directly (no-op Reshape avoided)
          newInputs.push_back(constValue);
        } else {
          // Shape changed - need Reshape
          auto newType =
              RankedTensorType::get(newShape, constType.getElementType());

          auto shapeType = RankedTensorType::get(
              {static_cast<int64_t>(newShape.size())}, rewriter.getI64Type());
          auto shapeAttr =
              DenseElementsAttr::get(shapeType, ArrayRef<int64_t>(newShape));
          auto shapeConst = rewriter.create<ONNXConstantOp>(op.getLoc(),
              shapeType, Attribute(), shapeAttr, nullptr, nullptr, nullptr,
              nullptr, nullptr, nullptr);

          auto allowzeroAttr =
              rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), 0);
          auto reshapeOp = rewriter.create<ONNXReshapeOp>(op.getLoc(), newType,
              constValue, shapeConst.getResult(), allowzeroAttr);

          newInputs.push_back(reshapeOp.getResult());
        }
      } else {
        // Need to transpose the constant data - skip for now as it's complex
        return failure();
      }
    }

    auto outputType = mlir::cast<RankedTensorType>(op.getType());
    auto invPerm = inversePermutation(firstPerm);
    auto newOutputShape = permuteShape(outputType.getShape(), invPerm);
    auto newOutputType =
        RankedTensorType::get(newOutputShape, outputType.getElementType());

    auto newOp =
        rewriter.create<VariadicOp>(op.getLoc(), newOutputType, newInputs);

    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(op, op.getType(),
        newOp.getResult(), rewriter.getI64ArrayAttr(firstPerm));

    return success();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ONNXTransposeOptimizationPass
    : public PassWrapper<ONNXTransposeOptimizationPass,
          OperationPass<func::FuncOp>> {

  StringRef getArgument() const override {
    return "onnx-transpose-optimization";
  }

  StringRef getDescription() const override {
    return "Fuse ONNX transpose operations with compatible operations";
  }

  ONNXTransposeOptimizationPass() = default;
  ONNXTransposeOptimizationPass(const ONNXTransposeOptimizationPass &pass)
      : PassWrapper<ONNXTransposeOptimizationPass,
            OperationPass<func::FuncOp>>() {}

  Option<unsigned> maxIterations{*this, "max-iterations",
      llvm::cl::desc("Maximum number of greedy rewrite iterations"),
      llvm::cl::init(10)};

  void runOnOperation() override {
    auto function = getOperation();
    MLIRContext *context = &getContext();

    LLVM_DEBUG(llvm::dbgs() << "=== Running ONNX Transpose Fusion Pass ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Function: " << function.getName() << "\n");

    // Count transpose ops before
    int transposeCountBefore = 0;
    function.walk([&](ONNXTransposeOp /*op*/) { transposeCountBefore++; });
    LLVM_DEBUG(llvm::dbgs()
               << "Transpose ops before: " << transposeCountBefore << "\n");

    RewritePatternSet patterns(context);

    patterns.add<EliminateIdentityTranspose>(context);
    patterns.add<FuseConsecutiveTransposes>(context);

    patterns.add<PushTransposeThroughUnaryOp<ONNXReluOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXSigmoidOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXTanhOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXSqrtOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXAbsOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXNegOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXExpOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXEluOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXLeakyReluOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXCastOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXFloorOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXCeilOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXSinOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXCosOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXIdentityOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXLogOp>>(context);
    patterns.add<PushTransposeThroughUnaryOp<ONNXGeluOp>>(context);

    patterns.add<PushTransposeThroughClip>(context);
    patterns.add<PushTransposeThroughHardSigmoid>(context);

    patterns.add<MoveTransposeThroughReshape>(context);

    patterns.add<PushTransposeThroughQDQ<ONNXQuantizeLinearOp>>(context);
    patterns.add<PushTransposeThroughQDQ<ONNXDequantizeLinearOp>>(context);
    patterns.add<PushTransposeThroughSCast>(context);
    patterns.add<PushTransposeThroughUnaryOp<XCOMPILERRequantizeOp>>(context);
    patterns.add<FoldConstDQTranspose>(context);

    patterns.add<FuseBinaryOpTransposes<ONNXAddOp>>(context);
    patterns.add<FuseBinaryOpTransposes<ONNXSubOp>>(context);
    patterns.add<FuseBinaryOpTransposes<ONNXMulOp>>(context);
    patterns.add<FuseBinaryOpTransposes<ONNXDivOp>>(context);
    patterns.add<FuseBinaryOpTransposes<ONNXPowOp>>(context);
    patterns.add<FuseBinaryOpTransposes<ONNXGreaterOp>>(context);

    patterns.add<FuseTransposeImmuneBinaryOp<ONNXAddOp>>(context);
    patterns.add<FuseTransposeImmuneBinaryOp<ONNXSubOp>>(context);
    patterns.add<FuseTransposeImmuneBinaryOp<ONNXMulOp>>(context);
    patterns.add<FuseTransposeImmuneBinaryOp<ONNXDivOp>>(context);
    patterns.add<FuseTransposeImmuneBinaryOp<ONNXPowOp>>(context);
    patterns.add<FuseTransposeImmuneBinaryOp<ONNXGreaterOp>>(context);
    patterns.add<FuseTransposeImmuneBinaryOp<ONNXPReluOp>>(context);

    patterns.add<PushTransposeThroughBinaryWithConst<ONNXAddOp>>(context);
    patterns.add<PushTransposeThroughBinaryWithConst<ONNXSubOp>>(context);
    patterns.add<PushTransposeThroughBinaryWithConst<ONNXMulOp>>(context);
    patterns.add<PushTransposeThroughBinaryWithConst<ONNXDivOp>>(context);
    patterns.add<PushTransposeThroughBinaryWithConst<ONNXPowOp>>(context);
    patterns.add<PushTransposeThroughBinaryWithConst<ONNXGreaterOp>>(context);
    patterns.add<PushTransposeThroughBinaryWithConst<ONNXPReluOp>>(context);

    patterns.add<PushTransposeThroughVariadicWithConst<ONNXMinOp>>(context);
    patterns.add<PushTransposeThroughVariadicWithConst<ONNXMaxOp>>(context);

    patterns.add<PushTransposeThroughWhere>(context);

    patterns.add<PushTransposeThroughAxisOp<ONNXPadOp>>(context);
    patterns.add<PushTransposeThroughAxisOp<ONNXSliceOp>>(context);
    patterns.add<PushTransposeThroughAxisOp<ONNXTileOp>>(context);
    patterns.add<PushTransposeThroughAxisOp<ONNXExpandOp>>(context);

    patterns.add<PushTransposeThroughAxisOp<ONNXSqueezeOp>>(context);
    patterns.add<PushTransposeThroughAxisOp<ONNXArgMaxOp>>(context);
    patterns.add<PushTransposeThroughAxisOp<ONNXSoftmaxOp>>(context);

    patterns.add<PushTransposeThroughAxisOp<ONNXReduceMeanOp>>(context);
    patterns.add<PushTransposeThroughAxisOp<ONNXReduceMaxOp>>(context);
    patterns.add<PushTransposeThroughAxisOp<ONNXReduceMinOp>>(context);
    patterns.add<PushTransposeThroughAxisOp<ONNXReduceSumOp>>(context);

    patterns.add<PushTransposeThroughConcat>(context);

    // Apply patterns with greedy rewrite
    GreedyRewriteConfig config;
    config.maxIterations = maxIterations;
    config.useTopDownTraversal = true;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(applyPatternsGreedily(function, std::move(patterns), config))) {
      LLVM_DEBUG(llvm::dbgs() << "Pattern application failed!\n");
      signalPassFailure();
      return;
    }

    // Count transpose ops after
    int transposeCountAfter = 0;
    function.walk([&](ONNXTransposeOp /*op*/) { transposeCountAfter++; });
    LLVM_DEBUG(
        llvm::dbgs() << "Transpose ops after: " << transposeCountAfter << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Eliminated/fused: "
                            << (transposeCountBefore - transposeCountAfter)
                            << " transposes\n");
    LLVM_DEBUG(
        llvm::dbgs() << "=== ONNX Transpose Fusion Pass Completed ===\n");
  }
};

} // namespace onnx_mlir

// Factory function to create the pass
std::unique_ptr<mlir::Pass> onnx_mlir::createONNXTransposeOptimizationPass() {
  return std::make_unique<onnx_mlir::ONNXTransposeOptimizationPass>();
}
