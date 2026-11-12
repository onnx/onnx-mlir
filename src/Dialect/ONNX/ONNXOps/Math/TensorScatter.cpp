/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ TensorScatter.cpp - ONNX Operations --------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect TensorScatter operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXTensorScatterOp::verify() {
  ONNXTensorScatterOpAdaptor operandAdaptor(*this);

  // 'mode' must be one of the values defined by the ONNX spec.
  StringRef mode = operandAdaptor.getMode();
  if (mode != "linear" && mode != "circular")
    return emitOpError(
        "'mode' must be 'linear' or 'circular', but got '" + mode + "'");

  Value pastCache = operandAdaptor.getPastCache();
  Value update = operandAdaptor.getUpdate();

  if (!hasShapeAndRank(pastCache) || !hasShapeAndRank(update))
    return success();

  auto pastCacheType = mlir::cast<ShapedType>(pastCache.getType());
  auto updateType = mlir::cast<ShapedType>(update.getType());

  // 'past_cache' and 'update' must have the same element type.
  if (pastCacheType.getElementType() != updateType.getElementType())
    return emitOpError(
        "'past_cache' and 'update' must have the same element type");

  int64_t rank = pastCacheType.getRank();

  // 'past_cache' and 'update' must have the same rank.
  if (updateType.getRank() != rank)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), update, updateType.getRank(),
        std::to_string(rank));

  // 'axis' attribute must be in the range [-rank, rank-1].
  int64_t axis = this->getAxis();
  if (axis < -rank || axis >= rank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axis,
        onnx_mlir::Diagnostic::Range<int64_t>(-rank, rank - 1));
  if (axis < 0)
    axis += rank;

  // 'past_cache' and 'update' must have the same shape, except along 'axis',
  // where 'update's size (the sequence length) must not exceed 'past_cache's
  // size (the max sequence length). Checked dimension by dimension whenever
  // both are statically known.
  ArrayRef<int64_t> pastCacheShape = pastCacheType.getShape();
  ArrayRef<int64_t> updateShape = updateType.getShape();
  for (int64_t i = 0; i < rank; ++i) {
    if (pastCacheShape[i] == ShapedType::kDynamic ||
        updateShape[i] == ShapedType::kDynamic)
      continue;
    if (i == axis) {
      if (updateShape[i] > pastCacheShape[i])
        return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
            *this->getOperation(), update, i, updateShape[i],
            "<= " + std::to_string(pastCacheShape[i]));
    } else if (updateShape[i] != pastCacheShape[i]) {
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), update, i, updateShape[i],
          std::to_string(pastCacheShape[i]));
    }
  }

  // 'write_indices', when present, must be a 1D tensor whose size is the
  // batch size, i.e. 'past_cache's dimension 0.
  Value writeIndices = operandAdaptor.getWriteIndices();
  if (!isNoneValue(writeIndices) && hasShapeAndRank(writeIndices)) {
    auto writeIndicesType = mlir::cast<ShapedType>(writeIndices.getType());
    int64_t writeIndicesRank = writeIndicesType.getRank();
    if (writeIndicesRank != 1)
      return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
          *this->getOperation(), writeIndices, writeIndicesRank, "1");

    int64_t batchSize = pastCacheShape[0];
    int64_t writeIndicesSize = writeIndicesType.getShape()[0];
    if (batchSize != ShapedType::kDynamic &&
        writeIndicesSize != ShapedType::kDynamic &&
        writeIndicesSize != batchSize)
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), writeIndices, 0, writeIndicesSize,
          std::to_string(batchSize));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

// 'present_cache' always has the same shape and element type as 'past_cache',
// regardless of 'update' and 'write_indices': TensorScatter models an
// in-place update to a fixed-size cache buffer.
LogicalResult ONNXTensorScatterOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}
