/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Dim.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Dim operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXDimOpShapeHelper::computeShape() {
  // Dim returns tensor<1xi64>
  return setOutputDimsFromLiterals({1});
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXDimOp::verify() {
  if (!hasShapeAndRank(this->getData()))
    return emitOpError("input must have shape and rank.");

  int64_t axis = this->getAxis();
  if ((axis < 0) || (axis >= getRank(this->getData().getType())))
    return emitOpError("attribute ")
           << ONNXDimOp::getAxisAttrName() << " value is " << axis
           << ", accepted range is [0, "
           << getRank(this->getData().getType()) - 1 << "].";
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXDimOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type elementType = IntegerType::get(getContext(), 64);
  ONNXDimOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXDimOp>;
} // namespace onnx_mlir
