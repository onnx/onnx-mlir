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
  if (failed(verifyResultElementTypeEqualsDtypeWithFallBackToInputType(
          *this, getDtype())))
    return failure();

  auto inputType = mlir::dyn_cast<ShapedType>(getInput().getType());
  if (inputType && inputType.hasRank() && inputType.getRank() != 2)
    return emitOpError("Input should have a rank of two");
  return success();
}

//===----------------------------------------------------------------------===//
// Type and Shape Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXEyeLikeOp::resultTypeInference() {
  return getResultTypeForShapeCopyingOp(*this, getDtype());
}

LogicalResult ONNXEyeLikeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();
  return inferShapeForUnaryOps(getOperation(),
      getMLIRTypeFromDtypeWithFallBackToInputType(*this, getDtype()));
}