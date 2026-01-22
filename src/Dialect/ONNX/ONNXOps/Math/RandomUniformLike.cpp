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
  return verifyResultElementTypeEqualsDtypeWithFallBackToInputType(
      *this, getDtype());
}

//===----------------------------------------------------------------------===//
// Shape + Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXRandomUniformLikeOp::resultTypeInference() {
  return getResultTypeForShapeCopyingOp(*this, getDtype());
}

LogicalResult ONNXRandomUniformLikeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();
  return inferShapeForUnaryOps(getOperation(),
      getMLIRTypeFromDtypeWithFallBackToInputType(*this, getDtype()));
}