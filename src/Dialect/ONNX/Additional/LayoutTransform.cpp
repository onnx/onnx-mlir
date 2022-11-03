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

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

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
    std::function<void(mlir::Region &)> doShapeInference) {
  ONNXLayoutTransformOp operandAdaptor(*this);
  if (!hasShapeAndRank(operandAdaptor.data()))
    return success();

  auto builder = mlir::Builder(getContext());
  Type resType = convertTensorTypeToTensorTypeWithONNXTensorEncoding(
      builder, data().getType(), target_layoutAttr());
  getResult().setType(resType);
  return success();
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

void ONNXLayoutTransformOp::build(OpBuilder &builder, OperationState &state,
    Value data, StringAttr targetLayoutAttr) {
  Type resType = convertTensorTypeToTensorTypeWithONNXTensorEncoding(
      builder, data.getType(), targetLayoutAttr);
  build(builder, state, resType, data, targetLayoutAttr);
}
