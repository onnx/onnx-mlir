/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ LayoutTransform.cpp - ONNX Operations -------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXLayoutTransformOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getData()))
    return success();

  Type elementType =
      getData().getType().dyn_cast<RankedTensorType>().getElementType();
  ONNXUnaryOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      elementType, getTargetLayoutAttr());
}

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//
LogicalResult ONNXLayoutTransformOp::verify() {
  if (auto dataType = getData().getType().dyn_cast<RankedTensorType>())
    if (auto outputType = getOutput().getType().dyn_cast<RankedTensorType>())
      return success(getShape(dataType) == getShape(outputType));
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
