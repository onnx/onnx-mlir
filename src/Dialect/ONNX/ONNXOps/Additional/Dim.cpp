/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Dim.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Dim operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXDimOp::verify() {
  // Input data must be ranked.
  if (!hasShapeAndRank(this->data()))
    return failure();
  // Axis must be in [0, rank -1].
  int64_t axis = this->axis();
  return failure((axis < 0) || (axis >= getRank(this->data().getType())));
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXDimOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  OpBuilder b(getContext());
  getResult().setType(RankedTensorType::get({1}, b.getI64Type()));
  return success();
}
