/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Identity.cpp - ONNX Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Identity operation.
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
LogicalResult ONNXIdentityOpShapeHelper::computeShape() {
  ONNXIdentityOpAdaptor operandAdaptor(operands);
  return setOutputDimsFromOperand(operandAdaptor.getInput());
}
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXIdentityOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();

  // Since identity set the output to the same as the input, don't use the shape
  // helper infrastructure here, especially because we may have to deal with Opt
  // or Seq types.
  getResult().setType(getInput().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXIdentityOp>;
} // namespace onnx_mlir
