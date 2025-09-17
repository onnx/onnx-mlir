/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ RandomUniform.cpp - ONNX Operation ----------------===//
//
// Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
// =============================================================================
//
// This file provides definition of ONNX dialect RandomUniform operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomUniformOp::verify() {
  return verifyElementTypeFromDtype(*this);
}

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXRandomUniformOp::resultTypeInference() {
  return {UnrankedTensorType::get(getResultElementTypeFromDtype(*this))};
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomUniformOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  ONNXRandomUniformOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      getResultElementTypeFromDtype(*this));
}
