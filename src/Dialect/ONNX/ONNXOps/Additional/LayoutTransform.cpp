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

ONNXOpShapeHelper *ONNXLayoutTransformOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnaryOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXLayoutTransformOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(data()))
    return success();

  Type elementType =
      data().getType().dyn_cast<RankedTensorType>().getElementType();
  ONNXUnaryOpShapeHelper shapeHelper(this->getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      elementType, target_layoutAttr());
}

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//
LogicalResult ONNXLayoutTransformOp::verify() {
  if (hasShapeAndRank(data()) && hasShapeAndRank(output())) {
    if (getShape(data().getType()) != getShape(output().getType()))
      return emitOpError("Input and output tensors must have the same shape");
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
