/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ .cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect  operation.
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
LogicalResult ONNXEyeLikeOpShapeHelper::computeShape() {
  ONNXEyeLikeOpAdaptor operandAdaptor(operands);
  DimsExpr outputDims;
  createIE->getShapeAsDims(operandAdaptor.getInput(), outputDims);
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXEyeLikeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(getInput().getType());
  Type elementType;
  if (getDtypeAttr()) {
    auto builder = OpBuilder(getContext());
    elementType = convertONNXTypeToMLIRType(
        builder, static_cast<onnx::TensorProto_DataType>(
                     getDtypeAttr().getValue().getSExtValue()));
  } else {
    elementType = inputType.getElementType();
  }

  ONNXEyeLikeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXEyeLikeOp>;
} // namespace onnx_mlir
