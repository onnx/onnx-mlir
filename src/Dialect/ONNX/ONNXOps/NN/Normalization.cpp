/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Normalization.cpp - ONNX Operations ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Normalization operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// BatchNormalizationInferenceMode
//===----------------------------------------------------------------------===//

LogicalResult ONNXBatchNormalizationInferenceModeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!X().getType().isa<RankedTensorType>() ||
      !scale().getType().isa<RankedTensorType>() ||
      !B().getType().isa<RankedTensorType>() ||
      !mean().getType().isa<RankedTensorType>() ||
      !var().getType().isa<RankedTensorType>())
    return success();

  auto inputTensorTy = X().getType().cast<RankedTensorType>();
  auto scaleTensorTy = scale().getType().cast<RankedTensorType>();
  auto biasTensorTy = B().getType().cast<RankedTensorType>();
  auto meanTensorTy = mean().getType().cast<RankedTensorType>();
  auto varianceTensorTy = var().getType().cast<RankedTensorType>();

  // Check whether the shapes of scale, bias, mean and variance are valid.
  // Operand's dimensions can be in the form of NxCxD1xD2x...xDn or N.
  // In case of N, C is assumed to be 1.
  // 2-D tensors are assumed to be of shape NxC
  // Shapes of scale, bias, mean and variance must be C.
  int64_t c = -1;
  if (inputTensorTy.getShape().size() == 1) {
    c = 1;
  } else if (inputTensorTy.getShape().size() >= 2) {
    c = (inputTensorTy.getShape()[1] != -1) ? inputTensorTy.getShape()[1] : -1;
  }

  if (c != -1) {
    auto s = scaleTensorTy.getShape();
    auto b = biasTensorTy.getShape();
    auto m = meanTensorTy.getShape();
    auto v = varianceTensorTy.getShape();

    if ((s.size() != 1) || (s[0] != -1 && s[0] != c))
      return emitError("Wrong rank for the scale");
    if ((b.size() != 1) || (b[0] != -1 && b[0] != c))
      return emitError("Wrong rank for the bias");
    if ((m.size() != 1) || (m[0] != -1 && m[0] != c))
      return emitError("Wrong rank for the mean");
    if ((v.size() != 1) || (v[0] != -1 && v[0] != c))
      return emitError("Wrong rank for the variance");
  }

  // The output tensor of the same shape as the input.
  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// InstanceNormalization
//===----------------------------------------------------------------------===//

LogicalResult ONNXInstanceNormalizationOp::verify() {
  ONNXInstanceNormalizationOpAdaptor operandAdaptor =
      ONNXInstanceNormalizationOpAdaptor(*this);
  // Get operands.
  auto input = operandAdaptor.input();
  auto scale = operandAdaptor.scale();
  auto B = operandAdaptor.B();

  // Check input.
  if (!hasShapeAndRank(input)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  auto inputType = input.getType().cast<ShapedType>();
  auto inputShape = inputType.getShape();
  auto inputElementType = inputType.getElementType();
  int64_t spatialRank = inputShape.size() - 2;
  // If ranked, verify ranks of inputs.
  if (spatialRank < 1)
    return emitOpError("Spatial rank must be strictly positive");

  // Check bias B.
  if (hasShapeAndRank(B)) {
    // Can check at this stage.
    auto bType = B.getType().cast<ShapedType>();
    auto bShape = bType.getShape();
    if (bShape.size() != 1)
      return emitOpError("Bias should have a rank of one");
    if (bShape[0] >= 0 && inputShape[1] >= 0 && bShape[0] != inputShape[1])
      return emitOpError(
          "Bias should have same dimension as the second dimension of input");
    if (bType.getElementType() != inputElementType)
      return emitOpError("Bias should have same element type as input");
  }

  // Check scale.
  if (hasShapeAndRank(scale)) {
    // Can check at this stage.
    auto scaleType = scale.getType().cast<ShapedType>();
    auto scaleShape = scaleType.getShape();
    if (scaleShape.size() != 1)
      return emitOpError("Scale should have a rank of one");
    if (scaleShape[0] >= 0 && inputShape[1] >= 0 &&
        scaleShape[0] != inputShape[1])
      return emitOpError(
          "Scale should have same dimension as the second dimension of input");
    if (scaleType.getElementType() != inputElementType)
      return emitOpError("Scale should have same element type as input");
  }

  return success();
}
