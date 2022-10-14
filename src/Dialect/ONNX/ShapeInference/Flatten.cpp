/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Flatten.cpp - Shape Inference for Flatten Op --------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX Flatten Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXFlattenOpShapeHelper::computeShape(
    ONNXFlattenOpAdaptor operandAdaptor) {
  // Get info about input operand.
  Value input = operandAdaptor.input();
  auto inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputType.getRank();
  int64_t axis = op->axis();
  assert(axis >= -inputRank && axis < inputRank && "Invalid inputRank");

  // Negative axis means values are counted from the opposite side.
  if (axis < 0)
    axis += inputRank;

  // Compute outputDims.
  DimsExpr outputDims = {LiteralIndexExpr(1), LiteralIndexExpr(1)};
  for (int64_t i = 0; i < axis; ++i) {
    if (inputShape[i] == -1) {
      outputDims[0] = QuestionmarkIndexExpr();
      break;
    }
    outputDims[0] = outputDims[0] * LiteralIndexExpr(inputShape[i]);
  }

  for (int64_t i = axis; i < inputRank; ++i) {
    if (inputShape[i] == -1) {
      outputDims[1] = QuestionmarkIndexExpr();
      break;
    }
    outputDims[1] = outputDims[1] * LiteralIndexExpr(inputShape[i]);
  }

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir
