/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Gather.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Gather operation.
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
LogicalResult ONNXGatherOpShapeHelper::computeShape() {
  // Read data and indices shapes as dim indices.
  ONNXGatherOpAdaptor operandAdaptor(operands);
  ONNXGatherOp gatherOp = llvm::cast<ONNXGatherOp>(op);
  DimsExpr dataDims, indicesDims;
  createIE->getShapeAsDims(operandAdaptor.getData(), dataDims);
  createIE->getShapeAsDims(operandAdaptor.getIndices(), indicesDims);

  int64_t dataRank = dataDims.size();
  int64_t axisIndex = gatherOp.getAxis();
  assert(axisIndex >= -dataRank && axisIndex < dataRank && "Invalid axisIndex");
  // Negative value means counting dimensions from the back.
  axisIndex = (axisIndex < 0) ? axisIndex + dataRank : axisIndex;

  // Output has rank of 'indicesRank + (dataRank - 1).
  // Output shape is constructed from 'input' by:
  //    replacing the dimension at 'axis' in 'input' by the shape of
  //    'indices'.
  DimsExpr outputDims;
  for (int i = 0; i < dataRank; ++i) {
    if (i == axisIndex)
      for (IndexExpr d : indicesDims)
        outputDims.emplace_back(d);
    else
      outputDims.emplace_back(dataDims[i]);
  }

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXGatherOp::verify() {
  ONNXGatherOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(getOperation()))
    return success();

  auto dataType =
      mlir::cast<RankedTensorType>(operandAdaptor.getData().getType());
  ArrayRef<int64_t> dataShape = dataType.getShape();
  int64_t dataRank = dataShape.size();
  int64_t axisValue = getAxis();

  // axis attribute must be in the range [-r,r-1], where r = rank(data).
  if (axisValue < -dataRank || axisValue >= dataRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisValue,
        onnx_mlir::Diagnostic::Range<int64_t>(-dataRank, dataRank - 1));

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXGatherOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getOperation()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getData().getType()).getElementType();
  ONNXGatherOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXGatherOp>;
} // namespace onnx_mlir
