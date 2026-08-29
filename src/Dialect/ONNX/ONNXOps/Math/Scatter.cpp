/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Scatter.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Scatter operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Scatter
//===----------------------------------------------------------------------===//

LogicalResult ONNXScatterOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// ScatterElements
//===----------------------------------------------------------------------===//

LogicalResult ONNXScatterElementsOp::verify() {
  ONNXScatterElementsOpAdaptor operandAdaptor(*this);

  // 'reduction' must be one of the values defined by the ONNX spec.
  StringRef reduction = operandAdaptor.getReduction();
  if (reduction != "none" && reduction != "add" && reduction != "mul" &&
      reduction != "max" && reduction != "min")
    return emitOpError("'reduction' must be one of 'none', 'add', 'mul', "
                       "'max', or 'min', but got '" +
                       reduction + "'");

  if (!hasShapeAndRank(getOperation()))
    return success();

  // Get operands and attributes.
  Value data = operandAdaptor.getData();
  Value indices = operandAdaptor.getIndices();
  Value updates = operandAdaptor.getUpdates();
  auto dataType = mlir::cast<ShapedType>(data.getType());
  auto indicesType = mlir::cast<ShapedType>(indices.getType());
  auto updatesType = mlir::cast<ShapedType>(updates.getType());
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  int64_t updatesRank = updatesType.getRank();
  int64_t axis = this->getAxis();

  // 'data' and 'updates' must have the same element type.
  if (dataType.getElementType() != updatesType.getElementType())
    return emitOpError("'data' and 'updates' must have the same element "
                       "type");

  // All inputs must have the same rank, and the rank must be strictly greater
  // than zero.
  if (dataRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), data, dataRank, "> 0");
  if (indicesRank != dataRank)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), indices, indicesRank, std::to_string(dataRank));
  if (updatesRank != dataRank) {
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), updates, updatesRank, std::to_string(dataRank));
  }

  // 'indices' and 'updates' must have the same shape, checked dimension by
  // dimension whenever both are statically known.
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  ArrayRef<int64_t> updatesShape = updatesType.getShape();
  for (int64_t i = 0; i < dataRank; ++i) {
    if (indicesShape[i] == ShapedType::kDynamic ||
        updatesShape[i] == ShapedType::kDynamic)
      continue;
    if (indicesShape[i] != updatesShape[i])
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), updates, i, updatesShape[i],
          std::to_string(indicesShape[i]));
  }

  // axis attribute must be in the range [-r,r-1], where r = rank(data).
  if (axis < -dataRank || axis >= dataRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axis,
        onnx_mlir::Diagnostic::Range<int64_t>(-dataRank, dataRank - 1));

  if (axis < 0)
    axis += dataRank;

  // All index values in 'indices' are expected to be within bounds [-s, s-1]
  // along axis of size s.
  ArrayRef<int64_t> dataShape = dataType.getShape();
  const int64_t dataDimAtAxis = dataShape[axis];
  if (dataDimAtAxis >= 0) {
    if (ElementsAttr valueAttribute =
            getElementAttributeFromONNXValue(indices)) {
      if (isElementAttrUninitializedDenseResource(valueAttribute)) {
        return success(); // Return success to allow the parsing of MLIR with
                          // elided attributes
      }
      for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
        int64_t index = value.getInt();
        printf("Value %lld %lld\n", dataDimAtAxis, index);
        if (index >= -dataDimAtAxis && index < dataDimAtAxis)
          continue;

        return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
            *this->getOperation(), "indices", index,
            onnx_mlir::Diagnostic::Range<int64_t>(
                -dataDimAtAxis, dataDimAtAxis - 1));
      }
    }
  }

  return success();
}

LogicalResult ONNXScatterElementsOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// ScatterND
//===----------------------------------------------------------------------===//

LogicalResult ONNXScatterNDOp::verify() {
  ONNXScatterNDOpAdaptor operandAdaptor(*this);

  // 'reduction' must be one of the values defined by the ONNX spec.
  StringRef reduction = operandAdaptor.getReduction();
  if (reduction != "none" && reduction != "add" && reduction != "mul" &&
      reduction != "max" && reduction != "min")
    return emitOpError("'reduction' must be one of 'none', 'add', 'mul', "
                       "'max', or 'min', but got '" +
                       reduction + "'");

  if (!hasShapeAndRank(getOperation()))
    return success();

  // Get operands and attributes.
  Value data = operandAdaptor.getData();
  Value indices = operandAdaptor.getIndices();
  Value updates = operandAdaptor.getUpdates();
  auto dataType = mlir::cast<ShapedType>(data.getType());
  auto indicesType = mlir::cast<ShapedType>(indices.getType());
  auto updatesType = mlir::cast<ShapedType>(updates.getType());
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  int64_t updatesRank = updatesType.getRank();

  // 'data' and 'updates' must have the same element type.
  if (dataType.getElementType() != updatesType.getElementType())
    return emitOpError("'data' and 'updates' must have the same element "
                       "type");

  // 'data' and 'indices' must have rank strictly greater than zero.
  if (dataRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), data, dataRank, "> 0");
  if (indicesRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), indices, indicesRank, "> 0");

  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  ArrayRef<int64_t> updatesShape = updatesType.getShape();
  int64_t indicesLastDim = indicesShape[indicesRank - 1];

  // The rank of 'updates' must be equal to:
  //    rank(data) + rank(indices) - indices.shape[-1] - 1.
  // Only checked when indices.shape[-1] is statically known: it may
  // legitimately be zero, not just dynamic, so this must not be guarded by
  // "indicesLastDim > 0" (that silently skipped the check, and the size
  // dependent code below, for a known indices.shape[-1] of 0).
  if (indicesLastDim != ShapedType::kDynamic) {
    int64_t expectedUpdatesRank = dataRank + indicesRank - indicesLastDim - 1;
    if (updatesRank != expectedUpdatesRank)
      return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
          *this->getOperation(), updates, updatesRank,
          std::to_string(expectedUpdatesRank));

    // The last dimension of the 'indices' shape can be at most equal to the
    // rank of 'data'.
    if (indicesLastDim > dataRank)
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), indices, indicesRank - 1, indicesLastDim,
          "<= " + std::to_string(dataRank));
  }

  // The constraints check following this point requires the input tensors shape
  // dimensions to be known, if they aren't delay the checks.
  if (llvm::any_of(indicesShape,
          [](int64_t idx) { return (idx == ShapedType::kDynamic); }))
    return success();
  if (llvm::any_of(updatesShape,
          [](int64_t idx) { return (idx == ShapedType::kDynamic); }))
    return success();

  // Let q = rank(indices). The first (q-1) dimensions of the 'updates' shape
  // must match the first (q-1) dimensions of the 'indices' shape.
  for (int64_t i = 0; i < indicesRank - 1; ++i) {
    assert(i < updatesRank && "i is out of bounds");
    if (updatesShape[i] != indicesShape[i])
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), updates, i, updatesShape[i],
          std::to_string(indicesShape[i]));
  }

  if (llvm::any_of(
          dataShape, [](int64_t idx) { return (idx == ShapedType::kDynamic); }))
    return success();

  // Let k = indices.shape[-1], r = rank(data), q = rank(indices). Check that
  // updates.shape[q:] matches data.shape[k:r-1].
  for (int64_t i = indicesLastDim, j = indicesRank - 1; i < dataRank;
       ++i, ++j) {
    assert(j < updatesRank && "j is out of bounds");
    if (updatesShape[j] != dataShape[i])
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), updates, j, updatesShape[j],
          std::to_string(dataShape[i]));
  }

  return success();
}

LogicalResult ONNXScatterNDOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}
