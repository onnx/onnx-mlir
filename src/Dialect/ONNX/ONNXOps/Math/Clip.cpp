/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- Clip.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Clip operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXClipOpShapeHelper::computeShape() {
  ONNXClipOpAdaptor operandAdaptor(operands);
  return setOutputDimsFromOperand(operandAdaptor.getInput());
}

template <>
LogicalResult ONNXClipV6OpShapeHelper::computeShape() {
  ONNXClipOpAdaptor operandAdaptor(operands);
  return setOutputDimsFromOperand(operandAdaptor.getInput());
}

template <>
LogicalResult ONNXClipV11OpShapeHelper::computeShape() {
  ONNXClipOpAdaptor operandAdaptor(operands);
  return setOutputDimsFromOperand(operandAdaptor.getInput());
}

template <>
LogicalResult ONNXClipV12OpShapeHelper::computeShape() {
  ONNXClipOpAdaptor operandAdaptor(operands);
  return setOutputDimsFromOperand(operandAdaptor.getInput());
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

template <typename T>
LogicalResult ONNXClipOpGeneralInferShapes(
    std::function<void(Region &)> doShapeInference, Value input, Value min,
    Value max, Operation *op) {
  // Look at input.
  if (!hasShapeAndRank(input))
    return success();
  RankedTensorType inputTy = input.getType().cast<RankedTensorType>();
  Type elementType = inputTy.getElementType();
  // Look at optional min.
  if (!min.getType().isa<NoneType>()) {
    // Has a min, make sure its of the right type.
    if (!hasShapeAndRank(min))
      return success();
    // And size.
    RankedTensorType minTy = min.getType().cast<RankedTensorType>();
    if (minTy.getElementType() != elementType)
      return op->emitError(
          "Element type mismatch between input and min tensors");
    if (minTy.getShape().size() != 0)
      return op->emitError("Min tensor ranked with nonzero size");
  }
  // Look at optional max
  if (!max.getType().isa<NoneType>()) {
    // Has a max, make sure its of the right type.
    if (!hasShapeAndRank(max))
      return success();
    // And size.
    RankedTensorType maxTy = max.getType().cast<RankedTensorType>();
    if (maxTy.getElementType() != elementType)
      return op->emitError(
          "Element type mismatch between input and max tensors");
    if (maxTy.getShape().size() != 0)
      return op->emitError("Min tensor ranked with nonzero size");
  }
  ONNXNonSpecificOpShapeHelper<T> shapeHelper(op, {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

LogicalResult ONNXClipOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return ONNXClipOpGeneralInferShapes<ONNXClipOp>(
      doShapeInference, getInput(), getMin(), getMax(), getOperation());
}

LogicalResult ONNXClipV6Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();
  RankedTensorType inputTy = getInput().getType().cast<RankedTensorType>();
  Type elementType = inputTy.getElementType();
  ONNXClipV6OpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

LogicalResult ONNXClipV11Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return ONNXClipOpGeneralInferShapes<ONNXClipV11Op>(
      doShapeInference, getInput(), getMin(), getMax(), getOperation());
}

LogicalResult ONNXClipV12Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return ONNXClipOpGeneralInferShapes<ONNXClipV12Op>(
      doShapeInference, getInput(), getMin(), getMax(), getOperation());
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXClipOp>;
template struct ONNXNonSpecificOpShapeHelper<ONNXClipV6Op>;
template struct ONNXNonSpecificOpShapeHelper<ONNXClipV11Op>;
template struct ONNXNonSpecificOpShapeHelper<ONNXClipV12Op>;
} // namespace onnx_mlir
