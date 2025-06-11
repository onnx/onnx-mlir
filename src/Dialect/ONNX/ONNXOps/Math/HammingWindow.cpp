/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ HammingWindow.cpp - ONNX Operations
//-------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect HammingWindow operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support for ONNXHammingWindowOpShapeHelper
//===----------------------------------------------------------------------===//
namespace onnx_mlir {

template <>
LogicalResult ONNXHammingWindowOpShapeHelper::computeShape() {

  ONNXHammingWindowOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  Value sizeTensor = operandAdaptor.getSize();
  IndexExpr dimSize = createIE->getIntAsSymbol(sizeTensor);

  DimsExpr outputDims;
  outputDims.emplace_back(dimSize);
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Shape Inference for the ONNXHammingWindowOp
//===----------------------------------------------------------------------===//

namespace mlir {
LogicalResult ONNXHammingWindowOp::inferShapes(
    std::function<void(mlir::Region &)> shapeInferenceFunc) {
  (void)shapeInferenceFunc;
  auto sizeType = getSize().getType().dyn_cast<RankedTensorType>();
  if (!sizeType || sizeType.getRank() != 0) {
    return success();
  }
  int64_t outputDataTypeInt = getOutputDatatype();
  Builder builder(getContext());
  Type elementType = onnx_mlir::convertONNXTypeToMLIRType(
      builder, static_cast<onnx::TensorProto_DataType>(outputDataTypeInt));
  onnx_mlir::ONNXHammingWindowOpShapeHelper shapeHelper(
      getOperation(), getOperand());
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

} // namespace mlir

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<mlir::ONNXHammingWindowOp>;

} // namespace onnx_mlir