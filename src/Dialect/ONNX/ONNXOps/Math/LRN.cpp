/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ LRNcpp - ONNX Operations --------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect LRN operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXLRNOpShapeHelper::computeShape() {
  ONNXLRNOpAdaptor operandAdaptor(operands);
  return setOutputDimsFromOperand(operandAdaptor.getX());
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXLRNOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type elementType = mlir::cast<ShapedType>(getX().getType()).getElementType();
  ONNXLRNOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXLRNOp>;
} // namespace onnx_mlir
