/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- ClipV6.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect ClipV6 operation.
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
LogicalResult ONNXClipV6OpShapeHelper::computeShape() {
  ONNXClipV6OpAdaptor operandAdaptor(operands);
  return setOutputDimsFromOperand(operandAdaptor.getInput());
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXClipV6Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Look at input.
  if (!hasShapeAndRank(getInput()))
    return success();
  RankedTensorType inputTy = getInput().getType().cast<RankedTensorType>();
  Type elementType = inputTy.getElementType();
  // Look at optional min.
  if (!getMin().getType().isa<NoneType>()) {
    // Has a min, make sure its of the right type.
    if (!hasShapeAndRank(getMin()))
      return success();
    // And size.
    RankedTensorType minTy = getMin().getType().cast<RankedTensorType>();
    if (minTy.getElementType() != elementType)
      return emitError("Element type mismatch between input and min tensors");
    if (minTy.getShape().size() != 0)
      return emitError("Min tensor ranked with nonzero size");
  }
  // Look at optional max
  if (!getMax().getType().isa<NoneType>()) {
    // Has a max, make sure its of the right type.
    if (!hasShapeAndRank(getMax()))
      return success();
    // And size.
    RankedTensorType maxTy = getMax().getType().cast<RankedTensorType>();
    if (maxTy.getElementType() != elementType)
      return emitError("Element type mismatch between input and max tensors");
    if (maxTy.getShape().size() != 0)
      return emitError("Min tensor ranked with nonzero size");
  }

  ONNXClipV6OpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXClipV6Op>;
} // namespace onnx_mlir
