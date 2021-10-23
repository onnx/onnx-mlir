/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- AveragePool.cpp - Shape Inference for AveragePool Op ---------===//
//
// This file implements shape inference for the ONNX AveragePool Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

LogicalResult ONNXAveragePoolOpShapeHelper::computeShape(
    ONNXAveragePoolOpAdaptor operandAdaptor) {
  return ONNXGenericPoolShapeHelper<ONNXAveragePoolOp,
      ONNXAveragePoolOpAdaptor>::computeShape(operandAdaptor, nullptr,
      op->kernel_shape(), op->pads(), op->strides(), None);
}
