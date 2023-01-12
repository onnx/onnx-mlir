/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Bernoulli.cpp - ONNX Operations -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
  if (!hasShapeAndRank(input())) {
    return success();
  }
  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  Type elementType;
  if (dtypeAttr()) {
    elementType = convertONNXTypeToMLIRType(builder,
        (onnx::TensorProto_DataType)dtypeAttr().getValue().getSExtValue());
  } else {
    elementType = inputType.getElementType();
  }
  getResult().setType(RankedTensorType::get(inputType.getShape(), elementType));
  return success();
}
