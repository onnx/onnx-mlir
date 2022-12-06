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

#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult NewONNXLRNOpShapeHelper::computeShape() {
  ONNXLRNOpAdaptor operandAdaptor(operands);
  return computeShapeFromOperand(operandAdaptor.X());
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXLRNOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto elementType = X().getType().cast<ShapedType>().getElementType();
  NewONNXLRNOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct NewONNXNonSpecificOpShapeHelper<ONNXLRNOp>;
} // namespace onnx_mlir
