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
  return verifyResultElementTypeEqualsDtype(*this, getDtype());
}

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXRandomNormalOp::resultTypeInference() {
  return {
      UnrankedTensorType::get(getMLIRTypeFromDtype(getContext(), getDtype()))};
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomNormalOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  ONNXRandomNormalOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      getMLIRTypeFromDtype(getContext(), getDtype()));
}
