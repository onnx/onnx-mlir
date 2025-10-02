/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ RandomUniformLike.cpp - ONNX Operations -----------===//
//
// Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
// =============================================================================
//
// This file provides definition of ONNX dialect RandomUniformLike operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomUniformLikeOp::verify() {
  return verifyElementTypeFromDtypeWithFallBackToInputType(*this);
}

//===----------------------------------------------------------------------===//
// Shape + Type Inference
//===----------------------------------------------------------------------===//

GET_SHAPE_AND_TYPE_INFERENCE_FOR_SHAPE_COPYING_OPS(ONNXRandomUniformLikeOp)