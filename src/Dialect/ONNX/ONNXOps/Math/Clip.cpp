/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Clip.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
  return computeShapeFromOperand(operandAdaptor.input());
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXClipOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Look at input.
  if (!input().getType().isa<RankedTensorType>())
    return success();
  RankedTensorType inputTy = input().getType().cast<RankedTensorType>();
  Type elementType = inputTy.getElementType();
  // Look at optional min.
  if (!min().getType().isa<NoneType>()) {
    // Has a min, make sure its of the right type.
    if (!min().getType().isa<RankedTensorType>())
      return success();
    // And size.
    RankedTensorType minTy = min().getType().cast<RankedTensorType>();
    if (minTy.getElementType() != elementType)
      return emitError("Element type mismatch between input and min tensors");
    if (minTy.getShape().size() != 0)
      return emitError("Min tensor ranked with nonzero size");
  }
  // Look at optional max
  if (!max().getType().isa<NoneType>()) {
    // Has a max, make sure its of the right type.
    if (!max().getType().isa<RankedTensorType>())
      return success();
    // And size.
    RankedTensorType maxTy = max().getType().cast<RankedTensorType>();
    if (maxTy.getElementType() != elementType)
      return emitError("Element type mismatch between input and max tensors");
    if (maxTy.getShape().size() != 0)
      return emitError("Min tensor ranked with nonzero size");
  }

  ONNXClipOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXClipOp>;
} // namespace onnx_mlir
