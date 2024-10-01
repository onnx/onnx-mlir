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

template <>
LogicalResult ONNXFlattenOpShapeHelper::computeShape() {
  // Get info about input operand.
  ONNXFlattenOpAdaptor operandAdaptor(operands);
  ONNXFlattenOp flattenOp = llvm::cast<ONNXFlattenOp>(op);
  Value input = operandAdaptor.getInput();
  auto inputType = mlir::cast<ShapedType>(input.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputType.getRank();
  int64_t axis = flattenOp.getAxis();
  assert(axis >= -inputRank && axis < inputRank && "Invalid inputRank");

  // Negative axis means values are counted from the opposite side.
  if (axis < 0)
    axis += inputRank;

  // Warning: code does appear to only work for shape inference.
  // Compute outputDims.
  DimsExpr outputDims = {LitIE(1), LitIE(1)};
  for (int64_t i = 0; i < axis; ++i) {
    if (ShapedType::isDynamic(inputShape[i])) {
      outputDims[0] = QuestionmarkIndexExpr(/*isFloat*/ false);
      break;
    }
    outputDims[0] = outputDims[0] * LitIE(inputShape[i]);
  }

  for (int64_t i = axis; i < inputRank; ++i) {
    if (ShapedType::isDynamic(inputShape[i])) {
      outputDims[1] = QuestionmarkIndexExpr(/*isFloat*/ false);
      break;
    }
    outputDims[1] = outputDims[1] * LitIE(inputShape[i]);
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
  if (!hasShapeAndRank(getInput()))
    return success();

  auto inputType = mlir::cast<ShapedType>(getInput().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  int64_t axisValue = getAxis();

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
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape is not yet known.
  if (!hasShapeAndRank(getInput()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getInput().getType()).getElementType();
  ONNXFlattenOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXFlattenOp>;
} // namespace onnx_mlir
