/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Flatten.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Flatten operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXFlattenOp::verify() {
  // Cannot verify constraints if the input shape is not yet known.
  if (!hasShapeAndRank(input()))
    return success();

  auto inputType = input().getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  int64_t axisValue = axis();

  // axis attribute must be in the range [-r,r], where r = rank(input).
  if (axisValue < -inputRank || axisValue > inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisValue,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank));

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXFlattenOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape is not yet known.
  if (!hasShapeAndRank(input()))
    return success();

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXFlattenOpShapeHelper, ONNXFlattenOp,
      ONNXFlattenOpAdaptor>(*this, elementType);
}
