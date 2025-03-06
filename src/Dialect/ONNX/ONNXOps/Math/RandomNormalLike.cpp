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
    const auto elementTypeID =
        static_cast<onnx::TensorProto_DataType>(*elementTypeIDDType);
    if (elementTypeID !=
            onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16 &&
        elementTypeID !=
            onnx::TensorProto_DataType::TensorProto_DataType_FLOAT &&
        elementTypeID !=
            onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE &&
        elementTypeID !=
            onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16) {
      return emitOpError("dtype not float16, float, double or bfloat16");
    }
    if (elementTypeID ==
            onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16 &&
        outputType != FloatType::getF16(getContext()))
      return emitOpError("output tensor does not match float16 dtype.");
    else if (elementTypeID ==
                 onnx::TensorProto_DataType::TensorProto_DataType_FLOAT &&
             outputType != FloatType::getF32(getContext()))
      return emitOpError("output tensor does not match float dtype.");
    else if (elementTypeID ==
                 onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE &&
             outputType != FloatType::getF64(getContext()))
      return emitOpError("output tensor does not match double dtype.");
    else if (elementTypeID ==
                 onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16 &&
             outputType != FloatType::getBF16(getContext()))
      return emitOpError("output tensor does not match bfloat16 dtype.");
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
