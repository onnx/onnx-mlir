//===- ONNXTransposeOptimizationPatterns.cpp - Generic Transpose Optimization
//--------===//
//
// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the generic patterns for optimizing transposes with
/// ONNX operations that have axis-dependent attributes.
///
//===----------------------------------------------------------------------===//

#include "ONNXTransposeOptimizationAxisChangePatterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Safely extract integer values from a DenseElementsAttr, handling different
/// integer bitwidths (i32, i64, etc.)
static SmallVector<int64_t> extractIntValues(DenseElementsAttr attr) {
  SmallVector<int64_t> result;
  if (!attr)
    return result;

  Type elementType = attr.getElementType();
  if (elementType.isInteger(64)) {
    for (auto val : attr.getValues<int64_t>())
      result.push_back(val);
  } else if (elementType.isInteger(32)) {
    for (auto val : attr.getValues<int32_t>())
      result.push_back(static_cast<int64_t>(val));
  } else if (elementType.isInteger(16)) {
    for (auto val : attr.getValues<int16_t>())
      result.push_back(static_cast<int64_t>(val));
  } else if (elementType.isInteger(8)) {
    for (auto val : attr.getValues<int8_t>())
      result.push_back(static_cast<int64_t>(val));
  } else {
    // Fallback: try using APInt which works for any integer type
    for (auto val : attr.getValues<llvm::APInt>())
      result.push_back(val.getSExtValue());
  }
  return result;
}

LogicalResult transformReductionAttributes(
    Operation *op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
  // Get the axes operand (second operand for reduction ops)
  Value axesOperand = op->getOperand(1);
  auto axesConstOp = axesOperand.getDefiningOp<ONNXConstantOp>();
  if (!axesConstOp)
    return failure();

  auto axesAttr =
      mlir::dyn_cast_or_null<DenseElementsAttr>(axesConstOp.getValueAttr());
  if (!axesAttr)
    return failure();

  // Extract current axes
  SmallVector<int64_t> axes = extractIntValues(axesAttr);

  // Transform axes according to permutation
  // When pushing transpose through: Transpose(input, perm) ->
  // Reduce(transposed, axes) becomes: Reduce(input, newAxes) ->
  // Transpose(result, ...) Each axis in transposed space corresponds to
  // perm[axis] in original space
  auto rank = static_cast<int64_t>(perm.size());
  SmallVector<int64_t> newAxes;
  for (int64_t axis : axes) {
    // Normalize negative axis: -1 means last dim, -2 means second-to-last, etc.
    if (axis < 0)
      axis += rank;

    if (axis < 0 || axis >= rank)
      return failure();

    newAxes.push_back(perm[axis]);
  }

  // Create new axes constant
  auto axesType = RankedTensorType::get(
      {static_cast<int64_t>(newAxes.size())}, rewriter.getI64Type());
  auto newAxesAttr =
      DenseElementsAttr::get(axesType, ArrayRef<int64_t>(newAxes));
  auto newAxesConst =
      rewriter.create<ONNXConstantOp>(op->getLoc(), axesType, Attribute(),
          newAxesAttr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

  op->setOperand(1, newAxesConst.getResult());
  return success();
}

SmallVector<int64_t> getReductionAdjustedPermutation(Operation *op,
    ArrayRef<int64_t> perm, ArrayRef<int64_t> /*inputShape*/,
    ArrayRef<int64_t> /*outputShape*/) {
  // Get keepdims attribute
  auto keepdimsAttr = op->getAttrOfType<IntegerAttr>("keepdims");
  if (!keepdimsAttr || keepdimsAttr.getValue().getSExtValue() != 0) {
    // keepdims=1 or missing: no rank change
    return SmallVector<int64_t>();
  }

  // Get the axes that were reduced
  Value axesOperand = op->getOperand(1);
  auto axesConstOp = axesOperand.getDefiningOp<ONNXConstantOp>();
  if (!axesConstOp)
    return SmallVector<int64_t>();

  auto axesAttr =
      mlir::dyn_cast_or_null<DenseElementsAttr>(axesConstOp.getValueAttr());
  if (!axesAttr)
    return SmallVector<int64_t>();

  SmallVector<int64_t> reducedAxes;
  auto rank = static_cast<int64_t>(perm.size());
  for (int64_t val : extractIntValues(axesAttr)) {
    int64_t axis = val;
    // Normalize negative axis
    if (axis < 0)
      axis += rank;
    if (axis >= 0 && axis < rank)
      reducedAxes.push_back(axis);
  }

  // Sort axes in descending order for proper removal
  llvm::sort(reducedAxes, std::greater<int64_t>());

  // Compute adjusted permutation by removing reduced axes
  SmallVector<int64_t> adjustedPerm;
  for (int64_t p : perm) {
    // Count how many axes less than p were removed
    int64_t offset = 0;
    for (int64_t axis : reducedAxes) {
      if (axis < p)
        offset++;
    }

    // Check if this axis is being reduced
    bool isReduced = false;
    for (int64_t axis : reducedAxes) {
      if (axis == p) {
        isReduced = true;
        break;
      }
    }

    if (!isReduced) {
      adjustedPerm.push_back(p - offset);
    }
  }

  return adjustedPerm;
}

//===----------------------------------------------------------------------===//
// ONNXPadOp Attribute Transformation
//===----------------------------------------------------------------------===//

LogicalResult AxisAttributeTransformer<ONNXPadOp>::transformAttributes(
    ONNXPadOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
  // Get pads constant
  auto padsConstOp = op.getPads().getDefiningOp<ONNXConstantOp>();
  if (!padsConstOp)
    return failure();

  auto padsAttr =
      mlir::dyn_cast_or_null<DenseElementsAttr>(padsConstOp.getValueAttr());
  if (!padsAttr)
    return failure();

  // Extract pads values
  SmallVector<int64_t> pads = extractIntValues(padsAttr);

  size_t rank = perm.size();
  if (pads.size() != 2 * rank)
    return failure();

  // Apply inverse permutation to transform pads from transposed space to
  // original space
  SmallVector<int64_t> invPerm = inversePermutation(perm);
  SmallVector<int64_t> newPads(2 * rank);
  for (size_t i = 0; i < rank; ++i) {
    newPads[i] = pads[invPerm[i]];
    newPads[i + rank] = pads[invPerm[i] + rank];
  }

  // Create new pads constant
  auto padsType = RankedTensorType::get(
      {2 * static_cast<int64_t>(rank)}, rewriter.getI64Type());
  auto newPadsAttr =
      DenseElementsAttr::get(padsType, ArrayRef<int64_t>(newPads));
  auto newPadsConst =
      rewriter.create<ONNXConstantOp>(op.getLoc(), padsType, Attribute(),
          newPadsAttr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

  op.getPadsMutable().assign(newPadsConst.getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// ONNXSliceOp Attribute Transformation
//===----------------------------------------------------------------------===//

LogicalResult AxisAttributeTransformer<ONNXSliceOp>::transformAttributes(
    ONNXSliceOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
  auto rank = static_cast<int64_t>(perm.size());

  // Validate permutation is not empty
  if (rank == 0)
    return failure();

  // Get axes
  auto axesConstOp = op.getAxes().getDefiningOp<ONNXConstantOp>();
  if (!axesConstOp)
    return failure();

  auto axesAttr =
      mlir::dyn_cast_or_null<DenseElementsAttr>(axesConstOp.getValueAttr());
  if (!axesAttr)
    return failure();

  SmallVector<int64_t> axes = extractIntValues(axesAttr);

  // Skip pushing transpose through Slice when slicing on a single axis that
  // is axis 0. This preserves Transpose→Slice(axis=0) patterns used in Q/K/V
  // head splitting to match golden xcompiler behavior.
  // Multi-axis slices (e.g. full-rank [0,1,2,3] for cropping) are still
  // optimized.
  if (axes.size() == 1) {
    int64_t normAxis = axes[0] < 0 ? axes[0] + rank : axes[0];
    if (normAxis == 0)
      return failure();
  }

  // Validate axes is not empty
  if (axes.empty())
    return failure();

  // Normalize negative axes and validate
  SmallVector<int64_t> normalizedAxes;
  llvm::DenseSet<int64_t> seenAxes; // Check for duplicates

  for (int64_t axis : axes) {
    if (axis < 0)
      axis += rank;

    // Out of bounds check
    if (axis < 0 || axis >= rank)
      return failure();

    // Duplicate axis check
    if (seenAxes.contains(axis))
      return failure(); // Duplicate axes not allowed

    seenAxes.insert(axis);
    normalizedAxes.push_back(axis);
  }

  // Check if this is the full-rank sequential pattern [0,1,2,3,...]
  bool isFullRankSequential = (axes.size() == static_cast<size_t>(rank));
  if (isFullRankSequential) {
    for (size_t i = 0; i < normalizedAxes.size(); ++i) {
      if (normalizedAxes[i] != static_cast<int64_t>(i)) {
        isFullRankSequential = false;
        break;
      }
    }
  }

  // Compute inverse permutation
  SmallVector<int64_t> invPerm = inversePermutation(perm);

  // Validate inverse permutation is correct size
  if (invPerm.size() != perm.size())
    return failure();

  auto makeConst = [&](ArrayRef<int64_t> values) -> Value {
    auto newType = RankedTensorType::get(
        {static_cast<int64_t>(values.size())}, rewriter.getI64Type());
    auto newAttr = DenseElementsAttr::get(newType, values);
    return rewriter
        .create<ONNXConstantOp>(op.getLoc(), newType, Attribute(), newAttr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)
        .getResult();
  };

  // Read starts, ends, steps - must be constants
  auto readI64Array = [](Value operand) -> std::optional<SmallVector<int64_t>> {
    if (!operand)
      return std::nullopt;
    auto constOp = operand.getDefiningOp<ONNXConstantOp>();
    if (!constOp)
      return std::nullopt;
    auto attr =
        mlir::dyn_cast_or_null<DenseElementsAttr>(constOp.getValueAttr());
    if (!attr)
      return std::nullopt;
    SmallVector<int64_t> values = extractIntValues(attr);
    return values;
  };

  auto starts = readI64Array(op.getStarts());
  auto ends = readI64Array(op.getEnds());
  auto steps = readI64Array(op.getSteps());

  // All arrays must be constant
  if (!starts || !ends || !steps)
    return failure();

  // All arrays must have same size as axes
  if (starts->size() != axes.size() || ends->size() != axes.size() ||
      steps->size() != axes.size())
    return failure();

  // Validate no steps are zero
  for (int64_t step : *steps) {
    if (step == 0)
      return failure(); // Zero step is invalid
  }

  if (isFullRankSequential) {
    // STRATEGY 1: Full-rank sequential axes [0,1,2,3,...]
    // - Keep axes unchanged as [0,1,2,3]
    // - Reorder starts/ends/steps using inverse permutation

    // Double-check size matches rank
    if (starts->size() != static_cast<size_t>(rank))
      return failure();

    SmallVector<int64_t> newStarts(rank);
    SmallVector<int64_t> newEnds(rank);
    SmallVector<int64_t> newSteps(rank);
    for (size_t i = 0; i < invPerm.size(); ++i) {
      // Bounds check before indexing
      if (invPerm[i] < 0 || static_cast<size_t>(invPerm[i]) >= starts->size())
        return failure();

      newStarts[i] = (*starts)[invPerm[i]];
      newEnds[i] = (*ends)[invPerm[i]];
      newSteps[i] = (*steps)[invPerm[i]];
    }

    // Normalize axes to [0,1,2,3] if it had negative values
    if (axes != normalizedAxes) {
      op.getAxesMutable().assign(makeConst(normalizedAxes));
    }

    op.getStartsMutable().assign(makeConst(newStarts));
    op.getEndsMutable().assign(makeConst(newEnds));
    op.getStepsMutable().assign(makeConst(newSteps));

  } else {
    // STRATEGY 2: Sparse or non-sequential axes
    // - Remap axes using inverse permutation: newAxes[i] =
    // inversePerm[oldAxes[i]]
    // - Keep starts/ends/steps unchanged

    SmallVector<int64_t> newAxes;
    newAxes.reserve(normalizedAxes.size());

    for (int64_t axis : normalizedAxes) {
      // Bounds check: axis must be valid index into invPerm
      if (axis < 0 || static_cast<size_t>(axis) >= invPerm.size())
        return failure();

      newAxes.push_back(invPerm[axis]);
    }

    op.getAxesMutable().assign(makeConst(newAxes));
    // starts/ends/steps remain unchanged (already validated to match axes size)
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ONNXExpandOp Attribute Transformation
//===----------------------------------------------------------------------===//

LogicalResult AxisAttributeTransformer<ONNXExpandOp>::transformAttributes(
    ONNXExpandOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
  auto shapeConstOp = op.getShape().getDefiningOp<ONNXConstantOp>();
  if (!shapeConstOp)
    return failure();

  auto shapeAttr =
      mlir::dyn_cast_or_null<DenseElementsAttr>(shapeConstOp.getValueAttr());
  if (!shapeAttr)
    return failure();

  SmallVector<int64_t> shape = extractIntValues(shapeAttr);

  if (shape.size() != perm.size())
    return failure();

  // Apply inverse permutation to transform shape from transposed space to
  // original space
  SmallVector<int64_t> invPerm = inversePermutation(perm);
  SmallVector<int64_t> newShape(perm.size());
  for (size_t i = 0; i < invPerm.size(); ++i)
    newShape[i] = shape[invPerm[i]];

  auto shapeType = RankedTensorType::get(
      {static_cast<int64_t>(newShape.size())}, rewriter.getI64Type());
  auto newShapeAttr =
      DenseElementsAttr::get(shapeType, ArrayRef<int64_t>(newShape));
  auto newShapeConst =
      rewriter.create<ONNXConstantOp>(op.getLoc(), shapeType, Attribute(),
          newShapeAttr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

  op.getShapeMutable().assign(newShapeConst.getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// ONNXTileOp Attribute Transformation
//===----------------------------------------------------------------------===//

LogicalResult AxisAttributeTransformer<ONNXTileOp>::transformAttributes(
    ONNXTileOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
  auto repeatsConstOp = op.getRepeats().getDefiningOp<ONNXConstantOp>();
  if (!repeatsConstOp)
    return failure();

  auto repeatsAttr =
      mlir::dyn_cast_or_null<DenseElementsAttr>(repeatsConstOp.getValueAttr());
  if (!repeatsAttr)
    return failure();

  SmallVector<int64_t> repeats = extractIntValues(repeatsAttr);

  if (repeats.size() != perm.size())
    return failure();

  // Apply inverse permutation to transform repeats from transposed space to
  // original space
  SmallVector<int64_t> invPerm = inversePermutation(perm);
  SmallVector<int64_t> newRepeats(perm.size());
  for (size_t i = 0; i < invPerm.size(); ++i)
    newRepeats[i] = repeats[invPerm[i]];

  auto repeatsType = RankedTensorType::get(
      {static_cast<int64_t>(newRepeats.size())}, rewriter.getI64Type());
  auto newRepeatsAttr =
      DenseElementsAttr::get(repeatsType, ArrayRef<int64_t>(newRepeats));
  auto newRepeatsConst =
      rewriter.create<ONNXConstantOp>(op.getLoc(), repeatsType, Attribute(),
          newRepeatsAttr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

  op.getRepeatsMutable().assign(newRepeatsConst.getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// ONNXSqueezeOp Attribute Transformation
//===----------------------------------------------------------------------===//

LogicalResult AxisAttributeTransformer<ONNXSqueezeOp>::transformAttributes(
    ONNXSqueezeOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
  // Get axes operand
  Value axesOperand = op.getAxes();
  if (!axesOperand)
    return failure();

  auto axesConstOp = axesOperand.getDefiningOp<ONNXConstantOp>();
  if (!axesConstOp)
    return failure();

  auto axesAttr =
      mlir::dyn_cast_or_null<DenseElementsAttr>(axesConstOp.getValueAttr());
  if (!axesAttr)
    return failure();

  // Extract current axes
  SmallVector<int64_t> axes = extractIntValues(axesAttr);

  // Transform axes according to permutation
  // When pushing transpose through: Transpose(input, perm) ->
  // Squeeze(transposed, axes) becomes: Squeeze(input, newAxes) ->
  // Transpose(result, ...) Each axis in transposed space corresponds to
  // perm[axis] in original space
  auto rank = static_cast<int64_t>(perm.size());
  SmallVector<int64_t> newAxes;
  for (int64_t axis : axes) {
    // Normalize negative axis: -1 means last dim, -2 means second-to-last, etc.
    if (axis < 0)
      axis += rank;

    if (axis < 0 || axis >= rank)
      return failure();

    newAxes.push_back(perm[axis]);
  }

  // Create new axes constant
  auto axesType = RankedTensorType::get(
      {static_cast<int64_t>(newAxes.size())}, rewriter.getI64Type());
  auto newAxesAttr =
      DenseElementsAttr::get(axesType, ArrayRef<int64_t>(newAxes));
  auto newAxesConst =
      rewriter.create<ONNXConstantOp>(op.getLoc(), axesType, Attribute(),
          newAxesAttr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

  op.getAxesMutable().assign(newAxesConst.getResult());
  return success();
}

SmallVector<int64_t>
AxisAttributeTransformer<ONNXSqueezeOp>::getAdjustedPermutation(
    ONNXSqueezeOp op, ArrayRef<int64_t> perm, ArrayRef<int64_t> /*inputShape*/,
    ArrayRef<int64_t> /*outputShape*/) {
  // Get the axes being squeezed
  Value axesOperand = op.getAxes();
  if (!axesOperand)
    return SmallVector<int64_t>();

  auto axesConstOp = axesOperand.getDefiningOp<ONNXConstantOp>();
  if (!axesConstOp)
    return SmallVector<int64_t>();

  auto axesAttr =
      mlir::dyn_cast_or_null<DenseElementsAttr>(axesConstOp.getValueAttr());
  if (!axesAttr)
    return SmallVector<int64_t>();

  SmallVector<int64_t> squeezedAxes;
  auto rank = static_cast<int64_t>(perm.size());
  for (int64_t val : extractIntValues(axesAttr)) {
    int64_t axis = val;
    // Normalize negative axis
    if (axis < 0)
      axis += rank;
    if (axis >= 0 && axis < rank)
      squeezedAxes.push_back(axis);
  }

  // Sort axes in descending order for proper removal
  llvm::sort(squeezedAxes, std::greater<int64_t>());

  // Compute adjusted permutation by removing squeezed axes
  SmallVector<int64_t> adjustedPerm;
  for (int64_t p : perm) {
    // Count how many axes less than p were removed
    int64_t offset = 0;
    for (int64_t axis : squeezedAxes) {
      if (axis < p)
        offset++;
    }

    // Check if this axis is being squeezed
    bool isSqueezed = false;
    for (int64_t axis : squeezedAxes) {
      if (axis == p) {
        isSqueezed = true;
        break;
      }
    }

    if (!isSqueezed) {
      adjustedPerm.push_back(p - offset);
    }
  }

  return adjustedPerm;
}

//===----------------------------------------------------------------------===//
// ONNXArgMaxOp Attribute Transformation
//===----------------------------------------------------------------------===//

LogicalResult AxisAttributeTransformer<ONNXArgMaxOp>::transformAttributes(
    ONNXArgMaxOp op, PatternRewriter &rewriter, ArrayRef<int64_t> perm) {
  auto axisAttr = op.getAxisAttr();
  if (!axisAttr)
    return failure();

  int64_t axis = axisAttr.getValue().getSExtValue();
  auto rank = static_cast<int64_t>(perm.size());

  // Normalize negative axis: -1 means last dim, -2 means second-to-last, etc.
  // Valid range: [-rank, rank-1]
  if (axis < 0)
    axis += rank;

  if (axis < 0 || axis >= rank)
    return failure();

  // When pushing transpose through: Transpose(input, perm) -> Op(transposed,
  // axis) becomes: Op(input, newAxis) -> Transpose(result, ...) The axis in
  // transposed space corresponds to perm[axis] in original space
  int64_t newAxis = perm[axis];

  // Create signed integer attribute for axis
  auto newAxisAttr = rewriter.getIntegerAttr(
      rewriter.getIntegerType(64, /*isSigned=*/true), newAxis);
  op.setAxisAttr(newAxisAttr);
  return success();
}

SmallVector<int64_t>
AxisAttributeTransformer<ONNXArgMaxOp>::getAdjustedPermutation(ONNXArgMaxOp op,
    ArrayRef<int64_t> perm, ArrayRef<int64_t> /*inputShape*/,
    ArrayRef<int64_t> /*outputShape*/) {
  // Check if keepdims is 0 (rank-changing case)
  auto keepdimsAttr = op.getKeepdimsAttr();
  if (!keepdimsAttr || keepdimsAttr.getValue().getSExtValue() != 0) {
    // keepdims=1: no rank change
    return SmallVector<int64_t>();
  }

  // Get the axis being reduced (after transformation, this is in original
  // space)
  auto axisAttr = op.getAxisAttr();
  if (!axisAttr)
    return SmallVector<int64_t>();

  int64_t axis = axisAttr.getValue().getSExtValue();
  auto rank = static_cast<int64_t>(perm.size());

  // Normalize negative axis
  if (axis < 0)
    axis += rank;

  if (axis < 0 || axis >= rank)
    return SmallVector<int64_t>();

  // Compute adjusted permutation by removing the reduced axis
  SmallVector<int64_t> adjustedPerm;
  for (int64_t p : perm) {
    if (p < axis) {
      adjustedPerm.push_back(p);
    } else if (p > axis) {
      adjustedPerm.push_back(p - 1);
    }
    // Skip p == axis (it's being removed)
  }

  return adjustedPerm;
}

} // namespace mlir
