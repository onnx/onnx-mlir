/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- MaxPoolSingleOut.cpp - Shape Inference for MaxPoolSingleOut Op ---===//
//
// This file implements shape inference for the ONNX MaxPool Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXMaxPoolSingleOutOpShapeHelper::computeShape(
    ONNXMaxPoolSingleOutOpAdaptor operandAdaptor) {
  return ONNXGenericPoolShapeHelper<ONNXMaxPoolSingleOutOp,
      ONNXMaxPoolSingleOutOpAdaptor>::computeShape(operandAdaptor, nullptr,
      op->kernel_shape(), op->pads(), op->strides(), op->dilations());
}

} // namespace onnx_mlir
