/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Unique.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Unique operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

LogicalResult ONNXUniqueOpShapeHelper::computeShape() {
  ONNXUniqueOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  // Get info about X and K operands.
  Value X = operandAdaptor.getX();
  if (!hasShapeAndRank(X)) {
    return failure();
  }
  int64_t rank = createIE->getShapedTypeRank(X);
  std::optional<int64_t> optionalAxis = operandAdaptor.getAxis();
  // Generate the output dims.
  DimsExpr outputDims;
  if (!optionalAxis.has_value()) {                         // if no axis given
    outputDims.emplace_back(QuestionmarkIndexExpr(false)); // return 1D array
  } else {                                                 // if axis given
    int64_t axis = optionalAxis.value();
    axis = (axis < 0) ? (rank + axis) : axis;
    for (int64_t i = 0; i < rank; i++) {
      outputDims.emplace_back((i == axis) ? QuestionmarkIndexExpr(false)
                                          : createIE->getShapeAsDim(X, i));
    }
  }
  setOutputDims(outputDims, 0);
  DimsExpr indexDims;
  indexDims.emplace_back(QuestionmarkIndexExpr(false));
  setOutputDims(indexDims, 1);
  setOutputDims(indexDims, 2);
  setOutputDims(indexDims, 3);
  return success();
}

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXUniqueOp::verify() {
  std::optional<int64_t> optionalSorted = getSorted();
  if (optionalSorted.has_value()) {
    // optional sorted attribute must be zero or one.
    int64_t sorted = optionalSorted.value();
    if (sorted < 0 || sorted > 1)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "sorted", sorted,
          onnx_mlir::Diagnostic::Range<int64_t>(0, 1));
  }
  ONNXUniqueOpAdaptor operandAdaptor(*this);
  Value X = operandAdaptor.getX();
  if (!hasShapeAndRank(X))
    return success(); // Too early to verify.

  // verify axis
  int64_t XRank = mlir::cast<ShapedType>(X.getType()).getRank();
  std::optional<int64_t> optionalAxis = getAxis();

  if (optionalAxis.has_value()) {
    // axis attribute must be in the range [-r,r-1], where r = rank(X).
    int64_t axis = optionalAxis.value();
    if (axis < -XRank || axis >= XRank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axis,
          onnx_mlir::Diagnostic::Range<int64_t>(-XRank, XRank - 1));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXUniqueOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Builder b = Builder(getContext());
  Type elementType = mlir::cast<ShapedType>(getX().getType()).getElementType();
  Type indexType = b.getI64Type();
  ONNXUniqueOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateTypes(
      {elementType, indexType, indexType, indexType});
}
