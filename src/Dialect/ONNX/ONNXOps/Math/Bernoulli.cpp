/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Bernoulli.cpp - ONNX Operations -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Bernoulli operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXBernoulliOp::verify() {
  return verifyResultElementTypeEqualsDtypeWithFallBackToInputType(
      *this, getDtype());
}

//===----------------------------------------------------------------------===//
// Type and Shape Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXBernoulliOp::resultTypeInference() {
  return getResultTypeForShapeCopyingOp(*this, getDtype());
}

LogicalResult ONNXBernoulliOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();
  return inferShapeForUnaryOps(getOperation(),
      getMLIRTypeFromDtypeWithFallBackToInputType(*this, getDtype()));
}