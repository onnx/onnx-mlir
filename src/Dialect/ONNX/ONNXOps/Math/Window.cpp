/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Window.cpp - ONNX Operations//--------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Windows operation.
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

template <typename OpTy>
LogicalResult ONNXWindowsOpShapeHelper<OpTy>::computeShape() {
  typename OpTy::Adaptor operandAdaptor(
      this->operands, this->op->getAttrDictionary());

  Value sizeTensor = operandAdaptor.getSize();
  IndexExpr dimSize = this->createIE->getIntAsSymbol(sizeTensor);

  DimsExpr outputDims;
  outputDims.emplace_back(dimSize);
  this->setOutputDims(outputDims);
  return success();
}

template <typename OpTy>
LogicalResult inferWindowOpShape(OpTy op) {
  auto sizeType = mlir::dyn_cast<RankedTensorType>(op.getSize().getType());
  if (!sizeType || sizeType.getRank() != 0)
    return success();

  int64_t outputDataTypeInt = op.getOutputDatatype();
  Builder builder(op.getContext());
  Type elementType = convertONNXTypeToMLIRType(
      builder, static_cast<onnx::TensorProto_DataType>(outputDataTypeInt));

  ONNXWindowsOpShapeHelper<OpTy> shapeHelper(
      op.getOperation(), op.getOperand());
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// HammingWindow
//===----------------------------------------------------------------------===//

namespace mlir {

LogicalResult ONNXHammingWindowOp::inferShapes(
    std::function<void(mlir::Region &)> shapeInferenceFunc) {
  (void)shapeInferenceFunc;
  return onnx_mlir::inferWindowOpShape(*this);
}

//===----------------------------------------------------------------------===//
// BlackmanWindow
//===----------------------------------------------------------------------===//

LogicalResult ONNXBlackmanWindowOp::inferShapes(
    std::function<void(mlir::Region &)> shapeInferenceFunc) {
  (void)shapeInferenceFunc;
  return onnx_mlir::inferWindowOpShape(*this);
}

} // namespace mlir

//===----------------------------------------------------------------------===//
// Template Instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXWindowsOpShapeHelper<mlir::ONNXHammingWindowOp>;
template struct ONNXWindowsOpShapeHelper<mlir::ONNXBlackmanWindowOp>;
} // namespace onnx_mlir