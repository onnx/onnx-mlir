/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ RandomUniform.cpp - ONNX Operations//------------===//
//
// Copyright 2025 The IBM Research Authors.
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
  return verifyResultElementTypeEqualsDtype(*this, getDtype());
}

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXRandomUniformOp::resultTypeInference() {
  return {
      UnrankedTensorType::get(getMLIRTypeFromDtype(getContext(), getDtype()))};
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomUniformOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  ONNXRandomUniformOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      getMLIRTypeFromDtype(getContext(), getDtype()));
}
