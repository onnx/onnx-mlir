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

  auto inputType =
      mlir::cast<RankedTensorType>(input.getType()).getElementType();
  auto outputType =
      mlir::cast<RankedTensorType>(output.getType()).getElementType();

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

static Type getRandomNormalLikeOutputElementType(ONNXRandomNormalLikeOp op) {
  auto inputType = mlir::cast<TensorType>(op.getInput().getType());
  Type elementType = inputType.getElementType();
  if (op.getDtypeAttr()) {
    auto builder = OpBuilder(op.getContext());
    elementType = convertONNXTypeToMLIRType(
        builder, static_cast<onnx::TensorProto_DataType>(
                     op.getDtypeAttr().getValue().getSExtValue()));
  }
  return elementType;
}

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXRandomNormalLikeOp::resultTypeInference() {
  Type elementType = getRandomNormalLikeOutputElementType(*this);
  std::vector<Type> resultTypes;
  if (auto rankedInputType =
          mlir::dyn_cast<RankedTensorType>(getInput().getType())) {
    resultTypes.push_back(rankedInputType.clone(elementType));
  } else {
    resultTypes.push_back(UnrankedTensorType::get(elementType));
  }
  return resultTypes;
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomNormalLikeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();
  Type elementType = getRandomNormalLikeOutputElementType(*this);
  return inferShapeForUnaryOps(getOperation(), elementType);
}
