/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- ShapeTransform.cpp - ONNX Operations ---------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXShapeTransformOpShapeHelper::computeShape() {
  ONNXShapeTransformOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXShapeTransformOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return success();
}

LogicalResult ONNXShapeTransformOp::verify() {
  ONNXShapeTransformOpAdaptor operandAdaptor(*this);

  // Get operands.
  Value input = operandAdaptor.getInput();
  Value output = getOutput();

  // Input must have static shape.
  auto inputType = dyn_cast<ShapedType>(input.getType());
  if (!(inputType && inputType.hasStaticShape()))
    return emitError("Does not support input with unknown dimensions");

  // Output must have static shape.
  auto outputType = dyn_cast<ShapedType>(output.getType());
  if (!(outputType && outputType.hasStaticShape()))
    return emitError("Does not support output with unknown dimensions");

  // Input and output must have the same number of elements.
  uint64_t elementsInput = 1;
  for (uint64_t d : inputType.getShape())
    elementsInput *= d;
  uint64_t elementsOutput = 1;
  for (uint64_t d : outputType.getShape())
    elementsOutput *= d;
  if (elementsInput != elementsOutput)
    return emitError(
        "The number of elements in the input and output mismatched");

  return success();
}
