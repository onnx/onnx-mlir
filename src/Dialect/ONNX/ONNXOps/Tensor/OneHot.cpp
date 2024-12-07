/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ OneHot.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect OneHot operation.
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

LogicalResult ONNXOneHotOpShapeHelper::computeShape() {
  ONNXOneHotOp oneHotOp = llvm::cast<ONNXOneHotOp>(op);
  ONNXOneHotOpAdaptor operandAdaptor(operands);
  Value indices = operandAdaptor.getIndices();
  if (!hasShapeAndRank(indices)) {
    return failure();
  }
  int64_t indicesRank = createIE->getShapedTypeRank(indices);

  // Axis is a required attribute and should have default value of -1.
  axis = oneHotOp.getAxis();
  if (axis < 0)
    axis += indicesRank + 1;
  assert(axis >= 0 && axis <= indicesRank && "tested in verify");

  Value depthValue = operandAdaptor.getDepth();
  bool depthIsFloat = isa<FloatType>(getElementType(depthValue.getType()));
  depth = depthIsFloat
              ? createIE->getFloatAsNonAffine(depthValue).convertToIndex()
              : createIE->getIntAsDim(depthValue);
  if (depth.isLiteral()) {
    if (depth.getLiteral() < 1)
      return op->emitError("OneHot depth must be greater than 1");
  }

  // Compute outputDims
  int outputRank = indicesRank + 1;
  DimsExpr outputDims(outputRank);
  for (auto i = 0; i < outputRank; i++) {
    if (i == axis) {
      outputDims[i] = depth;
    } else if (i < axis) {
      outputDims[i] = createIE->getShapeAsDim(indices, i);
    } else {
      outputDims[i] = createIE->getShapeAsDim(indices, i - 1);
    }
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXOneHotOp::verify() {
  ONNXOneHotOpAdaptor operandAdaptor = ONNXOneHotOpAdaptor(*this);
  // Check indices.
  Value indices = operandAdaptor.getIndices();
  if (hasShapeAndRank(indices)) {
    // Get rank.
    int64_t indicesRank = mlir::cast<ShapedType>(indices.getType()).getRank();
    // Verify axis.
    int64_t axisValue = getAxis();
    // Unusually, with a rank of 3, acceptable values are 0 (before first) to 3
    // (after last).
    if (axisValue < 0)
      axisValue += indicesRank + 1;
    if (!(axisValue >= 0 && axisValue <= indicesRank))
      return emitOpError("OneHot axis value is out of range");
  }
  // Check that values is a rank 2 with 2 elements
  Value values = operandAdaptor.getValues();
  if (hasShapeAndRank(values)) {
    ShapedType valuesShape = mlir::cast<ShapedType>(values.getType());
    if (valuesShape.getRank() != 1)
      return emitOpError("OneHot values must be 1D tensor");
    int64_t dim = valuesShape.getDimSize(0);
    if (dim >= 0 && dim != 2)
      return emitOpError("OneHot values must be 1D tensor with 2 elements");
  }
  // Depth is a scalar, check when its a tensor of rank 0 or 1.
  Value depth = operandAdaptor.getDepth();
  if (hasShapeAndRank(depth)) {
    ShapedType depthShape = mlir::cast<ShapedType>(depth.getType());
    if (depthShape.getRank() == 1) {
      int64_t dim = depthShape.getDimSize(0);
      if (dim >= 0 && dim != 1)
        return emitOpError("OneHot depth can be 1D tensor with 1 elements");
    } else {
      if (depthShape.getRank() > 1)
        return emitOpError("OneHot depth must be 0 or 1D tensor");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXOneHotOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getIndices()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getValues().getType()).getElementType();
  ONNXOneHotOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
