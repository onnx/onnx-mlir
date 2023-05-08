/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ .cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect  operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXCustomOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getOperation()))
    return success();

  std::optional<ArrayAttr> inputIndexAttrs = getInputsForInfer();
  // Use the first input because their element type should be same
  //int64_t inputIdx = (inputIndexAttrs.getValue()[0]).cast<IntegerAttr>().getInt();
  int64_t inputIdx = 0;
  Type elementType = getOutputElementType().value_or(getElementType(getInputs()[inputIdx].getType()));

  ValueRange operands = getInputs(); // All the inputs, not other parameter
  if (inputIndexAttrs.has_value()) {
    SmallVector<Value, 4> specifiedInputs;
    for(auto indexAttr : inputIndexAttrs.value()) {
      specifiedInputs.emplace_back(getInputs()[indexAttr.cast<IntegerAttr>().getInt()]);
    }
    operands = specifiedInputs;
  }
  if (getShapeInferPattern() == "SameAs") {
    ONNXUnaryOpShapeHelper shapeHelper(getOperation(), operands);
    return shapeHelper.computeShapeAndUpdateType(elementType);
  }
    
  // TODO: This does not appear to be implemented, set to return an error.
  // getResult().setType(getOperand().getType());
  // return success();
  return emitOpError(
      "op is not supported at this time. Please open an issue on "
      "https://github.com/onnx/onnx-mlir and/or consider contributing "
      "code. "
      "Error encountered in shape inference.");
}
