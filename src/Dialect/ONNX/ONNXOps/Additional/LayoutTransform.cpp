/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ LayoutTransform.cpp - ONNX Operations -------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect LayoutTransform operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXLayoutTransformOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getData()))
    return success();

  Type elementType =
      mlir::dyn_cast<RankedTensorType>(getData().getType()).getElementType();
  ONNXUnaryOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      elementType, getTargetLayoutAttr());
}

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//
LogicalResult ONNXLayoutTransformOp::verify() {
  if (auto dataType = mlir::dyn_cast<RankedTensorType>(getData().getType())) {
    if (auto outputType =
            mlir::dyn_cast<RankedTensorType>(getOutput().getType())) {
      for (int64_t i = 0; i < dataType.getRank(); ++i) {
        // Check if there is an unknown dimension in the dataShape and
        // outputShape. If there is an unknown dimension, we will return true.
        // If we know the dimension of dataShape and outputShape they should be
        // equal, if not then we return false.
        if (dataType.getShape()[i] == ShapedType::kDynamic ||
            outputType.getShape()[i] == ShapedType::kDynamic)
          return success();
        else if (dataType.getShape()[i] != outputType.getShape()[i])
          return emitOpError(
              "Input and output tensors must have the same shape");
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

void ONNXLayoutTransformOp::build(OpBuilder &builder, OperationState &state,
    Value data, Attribute targetLayoutAttr) {
  Type resType = convertTensorTypeToTensorTypeWithEncoding(
      data.getType(), targetLayoutAttr);
  build(builder, state, resType, data, targetLayoutAttr);
}
