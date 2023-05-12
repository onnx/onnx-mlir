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
    // No shape inference pattern provided. Do not know how to infer shape
    return success();
  }

  std::optional<ArrayAttr> inputIndexAttrs = getInputsForInfer();
  // Use the first input because their element type should be same
  int64_t inputIdx = 0;
  if (inputIndexAttrs.has_value())
    (inputIndexAttrs->getValue()[0]).cast<IntegerAttr>().getInt();

  Type elementType = getOutputElementType().value_or(
      getElementType(getInputs()[inputIdx].getType()));

  ONNXCustomOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
