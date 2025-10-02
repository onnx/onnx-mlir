/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ .cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect  operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXEyeLikeOp::verify() {
  return verifyElementTypeFromDtypeWithFallBackToInputType(*this);
}

//===----------------------------------------------------------------------===//
// Type and Shape Inference
//===----------------------------------------------------------------------===//

GET_SHAPE_AND_TYPE_INFERENCE_FOR_SHAPE_COPYING_OPS(ONNXEyeLikeOp)