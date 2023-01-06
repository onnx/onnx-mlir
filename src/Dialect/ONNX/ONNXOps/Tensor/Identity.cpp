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
// Verify
//===----------------------------------------------------------------------===//

// TODO:
//   Verify that matrix sizes are valid for multiplication and addition.
//   Take into account the dimensionality of the matrix.

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXIdentityOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnimplementedOpShapeHelper>(
      op, oper, ieb, scope);
}

LogicalResult ONNXIdentityOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  getResult().setType(getOperand().getType());
  return success();
}
