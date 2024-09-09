/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Bernoulli.cpp - ONNX Operations -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Bernoulli operation.
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

LogicalResult ONNXBernoulliOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto builder = OpBuilder(getContext());
  if (!hasShapeAndRank(getInput())) {
    return success();
  }
  Type elementType;
  if (getDtypeAttr()) {
    elementType = convertONNXTypeToMLIRType(
        builder, static_cast<onnx::TensorProto_DataType>(
                     getDtypeAttr().getValue().getSExtValue()));
  } else {
    elementType =
        mlir::cast<RankedTensorType>(getInput().getType()).getElementType();
  }
  ONNXBernoulliOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
