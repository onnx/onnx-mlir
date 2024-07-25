/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ .cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
// Type Inference
//===----------------------------------------------------------------------===//

Type ONNXEyeLikeOp::getResultElementType() {
  const auto inputType = cast<TensorType>(getInput().getType());
  if (getDtypeAttr()) {
    auto builder = OpBuilder(getContext());
    return convertONNXTypeToMLIRType(builder,
        (onnx::TensorProto_DataType)getDtypeAttr().getValue().getSExtValue());
  }
  return inputType.getElementType();
}

std::vector<Type> ONNXEyeLikeOp::resultTypeInference() {
  Type elementType = getResultElementType();
  std::vector<Type> resultTypes;

  if (auto rankedInputType = dyn_cast<RankedTensorType>(getInput().getType())) {
    resultTypes.push_back(rankedInputType.clone(elementType));
  } else {
    resultTypes.push_back(UnrankedTensorType::get(elementType));
  }
  return resultTypes;
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXEyeLikeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getInput()))
    return success();

  Type elementType = getResultElementType();
  ONNXEyeLikeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXEyeLikeOp>;
} // namespace onnx_mlir
