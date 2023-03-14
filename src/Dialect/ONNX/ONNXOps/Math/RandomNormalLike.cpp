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

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomNormalLikeOp::verify() {
  ONNXRandomNormalLikeOpAdaptor operandAdaptor(*this);
  Value input = operandAdaptor.getInput();
  if (!hasShapeAndRank(input))
    return success();
  Value output = this->getOutput();
  if (!hasShapeAndRank(output))
    return success();

  auto inputType = input.getType().cast<RankedTensorType>().getElementType();
  auto outputType = output.getType().cast<RankedTensorType>().getElementType();

  auto elementTypeIDDType = operandAdaptor.getDtype();
  if (elementTypeIDDType) {
    int64_t elementTypeID = elementTypeIDDType.value();
    if (elementTypeID < 0 || elementTypeID > 2) {
      return emitOpError("dtype not 0, 1 or 2.");
    }
    if (elementTypeID == 0 && outputType != FloatType::getF16(getContext()))
      return emitOpError("output tensor does match 0 dtype.");
    else if (elementTypeID == 1 &&
             outputType != FloatType::getF32(getContext()))
      return emitOpError("output tensor does match 1 dtype.");
    else if (elementTypeID == 2 &&
             outputType != FloatType::getF64(getContext()))
      return emitOpError("output tensor does match 2 dtype.");
  } else if (inputType != outputType) {
    return emitOpError("output and input element types do not match.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomNormalLikeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();
  auto inputType = getInput().getType().cast<RankedTensorType>();
  auto elementTypeIDDType = getDtype();

  // Default output tensor type in all cases is the input tensor type.
  Type elementType;
  if (!elementTypeIDDType) {
    elementType = inputType.getElementType();
  } else {
    int64_t elementTypeID = elementTypeIDDType.value();
    if (elementTypeID == 0)
      elementType = FloatType::getF16(getContext());
    else if (elementTypeID == 1)
      elementType = FloatType::getF32(getContext());
    else if (elementTypeID == 2)
      elementType = FloatType::getF64(getContext());
    else
      return emitError("dtype attribute is invalid (use: 0, 1 or 2)");
  }

  return inferShapeForUnaryOps(getOperation(), elementType);
}
