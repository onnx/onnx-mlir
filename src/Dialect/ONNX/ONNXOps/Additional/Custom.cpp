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
  // TODO: This does not appear to be implemented, set to return an error.
  // getResult().setType(getOperand().getType());
  // return success();
  return emitOpError(
      "op is not supported at this time. Please open an issue on "
      "https://github.com/onnx/onnx-mlir and/or consider contributing "
      "code. "
      "Error encountered in shape inference.");
}
