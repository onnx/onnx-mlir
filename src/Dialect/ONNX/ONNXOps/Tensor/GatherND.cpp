/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ GatherND.cpp - ONNX Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect GatherND operation.
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

LogicalResult ONNXGatherNDOpShapeHelper::computeShape(
    ONNXGatherNDOpAdaptor operandAdaptor) {
  Value data = operandAdaptor.data();
  Value indices = operandAdaptor.indices();
  MemRefBoundsIndexCapture dataBounds(data);
  MemRefBoundsIndexCapture indicesBounds(indices);
  DimsExpr dataDims, indicesDims;
  dataBounds.getDimList(dataDims);
  indicesBounds.getDimList(indicesDims);

  int64_t dataRank = dataDims.size();
  int64_t indicesRank = indicesDims.size();
  int64_t b = op->batch_dims();

  assert(indices.getType().isa<ShapedType>() && "Expecting a shaped type");
  auto indicesType = indices.getType().cast<ShapedType>();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  int64_t indicesLastDim = indicesShape[indicesRank - 1];
  int64_t outputRank = dataRank + indicesRank - indicesLastDim - 1 - b;

  // Ensure the operator contraints are statisfied.
  assert(dataRank >= 1 && "dataRank should be >= 1");
  assert(indicesRank >= 1 && "indicesRank should be >= 1");
  assert(b >= 0 && "batch_dim should not be negative");
  assert(b < std::min(dataRank, indicesRank) &&
         "batch_dims must be smaller than the min(dataRank, indicesRank)");
  assert((indicesLastDim >= 1 && indicesLastDim <= dataRank - b) &&
         "indices.shape[-1] must be in the range [1, dataRank - b]");

  // Save the first 'b' dimension of the shape of the 'indices' tensor.
  DimsExpr batchDims;
  for (int64_t i = 0; i < b; ++i)
    batchDims.emplace_back(indicesDims[i]);

  // output.shape = batchDims + list(indices.shape)[b:-1]
  DimsExpr outputDims;
  for (int64_t i = 0; i < b; ++i)
    outputDims.emplace_back(batchDims[i]);
  for (int64_t i = b; i < indicesRank - 1; ++i)
    outputDims.emplace_back(indicesDims[i]);

  // When indices.shape[-1] < data_rank - b,
  //   output_shape += list(data.shape)[batch_dims + indices.shape[-1]:]
  if (indicesLastDim < dataRank - b)
    for (int64_t i = b + indicesLastDim; i < dataRank; ++i)
      outputDims.emplace_back(dataDims[i]);

  assert((int64_t)outputDims.size() == outputRank &&
         "Incorrect shape computation");

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXGatherNDOp::verify() {
  ONNXGatherNDOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  // Get operands and attributes.
  Value data = operandAdaptor.data();
  Value indices = operandAdaptor.indices();
  auto dataType = data.getType().cast<ShapedType>();
  auto indicesType = indices.getType().cast<ShapedType>();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  int64_t b = batch_dims();

  // 'data' and 'indices' must have rank strictly greater than zero.
  if (dataRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), data, dataRank, "> 0");
  if (indicesRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), indices, indicesRank, "> 0");

  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  int64_t indicesLastDim = indicesShape[indicesRank - 1];

  // b must be smaller than min(rank(data), rank(indices).
  int64_t minDataAndIndicesRank = std::min(dataRank, indicesRank);
  if (b >= minDataAndIndicesRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "batch_dims", b,
        onnx_mlir::Diagnostic::Range<int64_t>(0, minDataAndIndicesRank - 1));

  // The first b dimensions of the shape of 'indices' and 'data' must be equal.
  for (int64_t i = 0; i < b; ++i) {
    int64_t dataDim = dataShape[i];
    int64_t indicesDim = indicesShape[i];
    if (indicesDim < 0 || dataDim < 0)
      continue;
    if (indicesDim != dataDim)
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), indices, i, indicesShape[i],
          std::to_string(dataShape[i]));
  }

  // Let r = rank(data), indices.shape[-1] must be in the range [1, r-b].
  if (indicesLastDim == 0)
    return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
        *this->getOperation(), indices, indicesRank - 1, indicesLastDim,
        ">= 1");
  if (indicesLastDim > dataRank - b)
    return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
        *this->getOperation(), indices, indicesRank - 1, indicesLastDim,
        "<= " + std::to_string(dataRank - b));

  // All values in 'indices' are expected to satisfy the inequality:
  //   -data.shape[b + i] <= indices[...,i] <= (data.shape[b + i]-1)].
  if (ElementsAttr valueAttribute = getElementAttributeFromONNXValue(indices)) {
    int flatIndex = 0;
    for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
      int64_t indexValue = value.getInt();
      int64_t gatherAxis = b + (flatIndex % indicesLastDim);
      int64_t dataDimAtAxis = dataShape[gatherAxis];
      if (dataDimAtAxis >= 0) {
        if (indexValue < -dataDimAtAxis || indexValue > dataDimAtAxis - 1)
          return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
              *this->getOperation(),
              "indices[" + std::to_string(flatIndex) + "]", indexValue,
              onnx_mlir::Diagnostic::Range<int64_t>(
                  -dataDimAtAxis, dataDimAtAxis - 1));
      }
      flatIndex++;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXGatherNDOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the shape of the output if the inputs shape is not yet known.
  if (llvm::any_of(
          this->getOperands(), [](Value op) { return !hasShapeAndRank(op); }))
    return success();

  // The output rank is given by:
  //   rank(output) = rank(indices) + rank(data) - indices_shape[-1] - 1 - b.
  // Therefore 'indices.shape[-1]' must be known in order to compute the output
  // shape.
  ArrayRef<int64_t> indicesShape =
      indices().getType().cast<ShapedType>().getShape();
  int64_t indicesRank = indicesShape.size();
  if (indicesShape[indicesRank - 1] < 0)
    return success(); // cannot infer the oputput shape yet.

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXGatherNDOpShapeHelper, ONNXGatherNDOp,
      ONNXGatherNDOpAdaptor>(*this, elementType);
  return success();
}
