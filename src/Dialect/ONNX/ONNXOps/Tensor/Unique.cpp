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
LogicalResult ONNXUniqueOpShapeHelper::computeShape() { //XXX WORKTODO
#if 0
  DimsExpr outputDims;
  ONNXUniqueOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  // Get info about X and K operands.
  Value X = operandAdaptor.X();
  Optional<int64_t> optionalAxis = operandAdaptor.axis();
  if (hasShapeAndRank(X)) {
    int64_t Xrank = X.getType().cast<ShapedType>().getRank();
    if (optionalAxis.has_value()) {
      // optional axis must be in the range [-Xrank, Xrank - 1].
      int64_t axis = optionalAxis.value();
      if (axis < -Xrank || axis >= Xrank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          operandAdaptor, "axis", axis,
          onnx_mlir::Diagnostic::Range<int64_t>(-Xrank, Xrank - 1));
    }
  }
  int64_t rank = createIE->getShapedTypeRank(X);

  // Axis to compute Unique
  int64_t flatten = !optionalAxis.has_value();
  int64_t axis = optionalAxis.has_value();

  // K is a scalar tensor storing the number of returned values along the given
  // axis.
  IndexExpr kIE = createIE->getIntAsSymbol(K);
  if (kIE.isUndefined())
    return op->emitError("K input parameter could not be processed");

  // If K is literal, it must be less than the axis dimension size.
  IndexExpr XAxisDim = createIE->getShapeAsDim(X, axis);
  if (kIE.isLiteral() && XAxisDim.isLiteral())
    if (kIE.getLiteral() >= XAxisDim.getLiteral())
      return op->emitError("K value is out of bound");

  for (int64_t i = 0; i < rank; ++i) {
    if (i == axis)
      outputDims.emplace_back(kIE);
    else
      outputDims.emplace_back(createIE->getShapeAsDim(X, i));
  }

  setOutputDims(outputDims, 0);
  setOutputDims(outputDims, 1);
#endif
  return success();
}


//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXUniqueOp::verify() {
  Optional<int64_t> optionalSorted = sorted();
  if (optionalSorted.has_value()) {
    // optional sorted attribute must be zero or one.
    int64_t sorted = optionalSorted.value();
    if (sorted < 0 || sorted > 1)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "sorted", sorted,
          onnx_mlir::Diagnostic::Range<int64_t>(0, 1));
  }
  ONNXUniqueOpAdaptor operandAdaptor(*this);
  Value X = operandAdaptor.X();
  if (!hasShapeAndRank(X))
    return success(); // Too early to verify.

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

