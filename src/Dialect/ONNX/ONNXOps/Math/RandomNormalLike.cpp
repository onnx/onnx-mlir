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
  return verifyResultElementTypeEqualsDtypeWithFallBackToInputType(
      *this, getDtype());
}

//===----------------------------------------------------------------------===//
// Shape + Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXRandomNormalLikeOp::resultTypeInference() {
  return getResultTypeForShapeCopyingOp(*this, getDtype());
}

LogicalResult ONNXRandomNormalLikeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();
  return inferShapeForUnaryOps(getOperation(),
      getMLIRTypeFromDtypeWithFallBackToInputType(*this, getDtype()));
}