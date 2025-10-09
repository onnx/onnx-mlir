/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ RandomNormal.cpp - ONNX Operations ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect RandomNormal operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomNormalOp::verify() {
  return verifyElementTypeFromDtype(*this);
}

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXRandomNormalOp::resultTypeInference() {
  return {UnrankedTensorType::get(getResultElementTypeFromDtype(*this))};
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomNormalOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  ONNXRandomNormalOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      getResultElementTypeFromDtype(*this));
}
