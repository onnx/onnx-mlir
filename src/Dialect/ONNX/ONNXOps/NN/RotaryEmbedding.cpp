// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.

// =============================================================================
//
// This file provides definition of ONNX dialect RotaryEmbedding operation.
//
//===----------------------------------------------------------------------===//

#include "ONNXOps.hpp"

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXRotaryEmbeddingOpShapeHelper::computeShape() {
  ONNXRotaryEmbeddingOpAdaptor operandAdaptor(operands);
  return setOutputDimsFromOperand(operandAdaptor.getX());
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRotaryEmbeddingOp::inferShapes(
    std::function<void(Region &)> /*shapeInferenceFunc*/) {
  // Cannot infer the output shape if the data input's shape is not yet known.
  if (!hasShapeAndRank(getOperation()->getOperand(0)))
    return success();

  Type elementType = mlir::cast<ShapedType>(getX().getType()).getElementType();
  ONNXRotaryEmbeddingOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXRotaryEmbeddingOp>;
} // namespace onnx_mlir