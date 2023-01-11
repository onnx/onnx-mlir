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

template <>
LogicalResult ONNXUniqueOpShapeHelper::computeShape() {
  ONNXUniqueOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  // Get info about X and K operands.
  Value X = operandAdaptor.X();
  int64_t rank = createIE->getShapedTypeRank(X);
  Optional<int64_t> optionalAxis = operandAdaptor.axis();
  // Generate the output dims.
  DimsExpr outputDims;
  LiteralIndexExpr minusone(-1);
  if (!optionalAxis.has_value()) {     // if no axis given
    outputDims.emplace_back(minusone); // return 1D array
  } else {                             // if axis given
    int64_t axis = optionalAxis.value();
    for (int64_t i = 0; i < rank; i++) {
      LiteralIndexExpr dim =
          (i == axis) ? minusone : createIE->getShapeAsDim(X, i);
      outputDims.emplace_back(dim);
    }
  }
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXUniqueOp::verify() {
  ONNXUniqueOpAdaptor operandAdaptor(*this);

  // verify X
  Value X = operandAdaptor.X();
  if (!hasShapeAndRank(X))
    return success(); // Too early to verify.

  // verify axis
  int64_t XRank = X.getType().cast<ShapedType>().getRank();
  Optional<int64_t> optionalAxis = axis();
  if (optionalAxis.has_value()) {
    // axis attribute must be in the range [-r,r-1], where r = rank(X).
    int64_t axis = optionalAxis.value();
    if (axis < -XRank || axis >= XRank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axis,
          onnx_mlir::Diagnostic::Range<int64_t>(-XRank, XRank - 1));
  }

  // verify sorted
  Optional<int64_t> optionalSorted = sorted();
  if (optionalSorted.has_value()) {
    // optional sorted attribute must be zero or one.
    int64_t sorted = optionalSorted.value();
    if (sorted < 0 || sorted > 1)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "sorted", sorted,
          onnx_mlir::Diagnostic::Range<int64_t>(0, 1));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXUniqueOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto builder = Builder(getContext());
  Type xType = getOperand().getType();
  if (!xType.isa<RankedTensorType>())
    return success();

  Type elementType = X().getType().cast<ShapedType>().getElementType();
  ONNXUniqueOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXUniqueOp>;
} // namespace onnx_mlir
