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

  // Get values.
  IndexExpr start = createIE->getIntFromArrayAsDim(operandAdaptor.start(), 0);
  IndexExpr limit = createIE->getIntFromArrayAsDim(operandAdaptor.limit(), 0);
  IndexExpr delta = createIE->getIntFromArrayAsDim(operandAdaptor.delta(), 0);
  // Dim = max(ceil((limit-start)/delta), 0).
  IndexExpr num = limit - start;
  num.ceilDiv(delta);
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
  if (!start().getType().isa<RankedTensorType>())
    return success();
  if (!limit().getType().isa<RankedTensorType>())
    return success();
  if (!delta().getType().isa<RankedTensorType>())
    return success();

  auto startTensorTy = start().getType().cast<RankedTensorType>();
  auto limitTensorTy = limit().getType().cast<RankedTensorType>();
  auto deltaTensorTy = delta().getType().cast<RankedTensorType>();

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

  if (!hasShapeAndRank(start()))
    return success();
  if (!hasShapeAndRank(limit()))
    return success();
  if (!hasShapeAndRank(delta()))
    return success();

  Type elementType =
      start().getType().cast<RankedTensorType>().getElementType();
  ONNXRangeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXRangeOp>;
} // namespace onnx_mlir
