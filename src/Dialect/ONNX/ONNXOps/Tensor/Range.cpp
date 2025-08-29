/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Range.cpp - ONNX Operations -----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Range operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXRangeOpShapeHelper::computeShape() {
  ONNXRangeOpAdaptor operandAdaptor(operands);

  Value startVal = operandAdaptor.getStart();
  Value limitVal = operandAdaptor.getLimit();
  Value deltaVal = operandAdaptor.getDelta();

  bool isFloat = isa<FloatType>(getElementType(startVal.getType()));

  // Calculate num = ceil((limit-start)/delta).
  IndexExpr num, start, limit, range, delta;

  ONNXAddOp addOp = limitVal.getDefiningOp<ONNXAddOp>();
  ONNXConstantOp cstOp;
  if (addOp && isDenseONNXConstant(deltaVal) &&
      matchValueAndOp<ONNXConstantOp>(
          addOp.getA(), addOp.getB(), startVal, cstOp)) {
    // Special case: "limit = start + range", where range and delta are scalar
    // constants. In this case "(limit-start)/delta" becomes "range/delta" which
    // is a constant.
    Value rangeVal = cstOp.getOutput();
    if (isFloat) {
      range = createIE->getFloatFromArrayAsNonAffine(rangeVal, 0);
      delta = createIE->getFloatFromArrayAsNonAffine(deltaVal, 0);
      num = (range / delta).ceil().convertToIndex();
    } else {
      range = createIE->getIntFromArrayAsDim(rangeVal, 0);
      delta = createIE->getIntFromArrayAsDim(deltaVal, 0);
      num = (range).ceilDiv(delta);
    }
  } else {
    // Normal case.
    if (isFloat) {
      start = createIE->getFloatFromArrayAsNonAffine(startVal, 0);
      limit = createIE->getFloatFromArrayAsNonAffine(limitVal, 0);
      delta = createIE->getFloatFromArrayAsNonAffine(deltaVal, 0);
      num = ((limit - start) / delta).ceil().convertToIndex();
    } else {
      start = createIE->getIntFromArrayAsDim(startVal, 0);
      limit = createIE->getIntFromArrayAsDim(limitVal, 0);
      delta = createIE->getIntFromArrayAsDim(deltaVal, 0);
      num = (limit - start).ceilDiv(delta);
    }
  }

  // Dim = max(ceil((limit-start)/delta), 0).
  IndexExpr res = IndexExpr::max(num, 0);
  DimsExpr outputDims(1, res);
  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXRangeOp::verify() {
  // All inputs must be valid ranked tensors.
  if (!hasShapeAndRank(getStart()) || !hasShapeAndRank(getLimit()) ||
      !hasShapeAndRank(getDelta()))
    return success();

  auto startTensorTy = mlir::cast<RankedTensorType>(getStart().getType());
  auto limitTensorTy = mlir::cast<RankedTensorType>(getLimit().getType());
  auto deltaTensorTy = mlir::cast<RankedTensorType>(getDelta().getType());

  // Only rank 0 or 1 input tensors are supported.
  if (startTensorTy.getShape().size() > 1)
    return emitError("start tensor must have rank zero or one");
  if (limitTensorTy.getShape().size() > 1)
    return emitError("limit tensor must have rank zero or one");
  if (deltaTensorTy.getShape().size() > 1)
    return emitError("delta tensor must have rank zero or one");

  // If tensor is rank 1 then the dimension has to be 1.
  if (startTensorTy.getShape().size() == 1 && startTensorTy.getShape()[0] > 1)
    return emitError("start tensor of rank one must have size one");
  if (limitTensorTy.getShape().size() == 1 && limitTensorTy.getShape()[0] > 1)
    return emitError("limit tensor of rank one must have size one");
  if (deltaTensorTy.getShape().size() == 1 && deltaTensorTy.getShape()[0] > 1)
    return emitError("delta tensor of rank one must have size one");

  // Only int or float input types are supported:
  // tensor(float), tensor(double), tensor(int16), tensor(int32),
  // tensor(int64)
  if (!startTensorTy.getElementType().isIntOrFloat())
    return emitError("start tensor type is not int or float");
  if (!limitTensorTy.getElementType().isIntOrFloat())
    return emitError("limit tensor type is not int or float");
  if (!deltaTensorTy.getElementType().isIntOrFloat())
    return emitError("delta tensor type is not int or float");

  // Additional condition for simplicity, enforce that all inputs have the
  // exact same element type:
  if (startTensorTy.getElementType() != limitTensorTy.getElementType() ||
      startTensorTy.getElementType() != deltaTensorTy.getElementType())
    return emitError("all inputs must have the exact same input type");

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRangeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // All inputs must be valid ranked tensors.

  if (!hasShapeAndRank(getStart()))
    return success();
  if (!hasShapeAndRank(getLimit()))
    return success();
  if (!hasShapeAndRank(getDelta()))
    return success();

  Type elementType =
      mlir::cast<RankedTensorType>(getStart().getType()).getElementType();
  ONNXRangeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXRangeOp>;
} // namespace onnx_mlir
