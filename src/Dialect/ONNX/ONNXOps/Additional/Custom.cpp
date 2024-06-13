/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ .cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Custom operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXCustomOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // ToDo: this check could be refined to the shape related input,
  // if inputs_for_infer is specified
  if (!hasShapeAndRank(getOperation()))
    return success();

  if (!getShapeInferPattern().has_value()) {
    // When no shape inference pattern provided, Just return.
    return success();
  } else if (getResults().size() > 1) {
    // ToFix: implementation limitation of existing ShapeHelper
    return emitError(
        "Shape inference pattern for multiple outputs NOT supported");
  }

  // Deterimine the element type of output.
  // Use output_element_type attribute if specified.
  // Otherwise,  use the first input in the list of inputs_for_infer.
  std::optional<ArrayAttr> inputIndexAttrs = getInputsForInfer();
  int64_t inputIdx = 0;
  if (inputIndexAttrs.has_value())
    inputIdx = mlir::cast<IntegerAttr>(inputIndexAttrs->getValue()[0]).getInt();

  Type elementType = getOutputElementType().value_or(
      getElementType(getInputs()[inputIdx].getType()));

  ONNXCustomOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
