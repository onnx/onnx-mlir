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

  ONNXClipV6OpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXClipV6Op>;
} // namespace onnx_mlir
