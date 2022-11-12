/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ GatherElements.cpp - ONNX Operations --------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect GatherElements operation.
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

LogicalResult ONNXGatherElementsOpShapeHelper::computeShape(
    ONNXGatherElementsOpAdaptor operandAdaptor) {
  MemRefBoundsIndexCapture indicesBounds(operandAdaptor.indices());
  DimsExpr indicesDims;
  indicesBounds.getDimList(indicesDims);

  // Output has the shape of indices.
  DimsExpr outputDims;
  int64_t indicesRank = indicesDims.size();
  for (int i = 0; i < indicesRank; ++i)
    outputDims.emplace_back(indicesDims[i]);

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXGatherElementsOp::verify() {
  ONNXGatherElementsOpAdaptor operandAdaptor(*this);
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
  int64_t axis = this->axis();

  // All inputs must have the same rank, and the rank must be strictly greater
  // than zero.
  if (dataRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), data, dataRank, "> 0");
  if (indicesRank != dataRank)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), indices, indicesRank, std::to_string(dataRank));

  // axis attribute must be in the range [-r,r-1], where r = rank(data).
  if (axis < -dataRank || axis >= dataRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axis,
        onnx_mlir::Diagnostic::Range<int64_t>(-dataRank, dataRank - 1));
  if (axis < 0)
    axis += dataRank;

  // All index values in 'indices' are expected to be within bounds [-s, s-1]
  // along axis of size s.
  ArrayRef<int64_t> dataShape = dataType.getShape();
  const int64_t dataDimAtAxis = dataShape[axis];
  if (dataDimAtAxis >= 0)
    if (ElementsAttr valueAttribute = getElementAttributeFromONNXValue(indices))
      for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
        int64_t index = value.getInt();
        if (index >= -dataDimAtAxis && index < dataDimAtAxis)
          continue;

        return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
            *this->getOperation(), "indices", index,
            onnx_mlir::Diagnostic::Range<int64_t>(
                -dataDimAtAxis, dataDimAtAxis - 1));
      }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXGatherElementsOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the output shape if the operands shape is not yet known.
  if (llvm::any_of(this->getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success();

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXGatherElementsOpShapeHelper,
      ONNXGatherElementsOp, ONNXGatherElementsOpAdaptor>(*this, elementType);
}
