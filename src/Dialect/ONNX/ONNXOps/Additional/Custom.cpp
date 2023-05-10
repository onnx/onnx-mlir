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
  if (!hasShapeAndRank(getOperation()))
    return success();

  std::optional<ArrayAttr> inputIndexAttrs = getInputsForInfer();
  // Use the first input because their element type should be same
  // int64_t inputIdx =
  // (inputIndexAttrs.getValue()[0]).cast<IntegerAttr>().getInt();
  int64_t inputIdx = 0;
  if (inputIndexAttrs.has_value())
    (inputIndexAttrs->getValue()[0]).cast<IntegerAttr>().getInt();

  Type elementType = getOutputElementType().value_or(
      getElementType(getInputs()[inputIdx].getType()));

  ValueRange operands;
  SmallVector<Value, 4> specifiedInputs;
  if (inputIndexAttrs.has_value()) {
    for (auto indexAttr : inputIndexAttrs.value()) {
      specifiedInputs.emplace_back(
          getInputs()[indexAttr.cast<IntegerAttr>().getInt()]);
    }
    operands = specifiedInputs;
  }
  if (!getShapeInferPattern().has_value()) {
    // No shape inference pattern provided. Simply return.
    return success();
  } else if (getShapeInferPattern() == "SameAs") {
    ONNXUnaryOpShapeHelper shapeHelper(getOperation(), operands);
    return shapeHelper.computeShapeAndUpdateType(elementType);
  } else if (getShapeInferPattern() == "MDBroadcast") {
    ONNXBroadcastOpShapeHelper shapeHelper(getOperation(), operands);
    return shapeHelper.computeShapeAndUpdateType(elementType);
  } else {
    return emitOpError("The specified shape_infer_pattern is not supported"
                       "Error encountered in shape inference.");
  }
}
