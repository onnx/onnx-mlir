/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ RandomNormalLike.cpp - ONNX Operations ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect RandomNormalLike operation.
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

LogicalResult ONNXRandomNormalLikeOp::verify() {
  return verifyElementTypeFromDtypeWithFallBackToInputType(*this);
}

//===----------------------------------------------------------------------===//
// Shape + Type Inference
//===----------------------------------------------------------------------===//

GET_SHAPE_AND_TYPE_INFERENCE_FOR_SHAPE_COPYING_OPS(ONNXRandomNormalLikeOp)