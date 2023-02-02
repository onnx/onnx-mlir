/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Reshape.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Reshape operation.
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
LogicalResult ONNXReshapeOpShapeHelper::computeShape() {
  ONNXReshapeOpAdaptor operandAdaptor(operands);
  DimsExpr outputDims;

  // Get info about input data operand.
  Value data = operandAdaptor.getData();
  int64_t dataRank = data.getType().cast<ShapedType>().getShape().size();

  // Get info about shape operand.
  Value shape = operandAdaptor.getShape();
  int64_t outputRank = createIE->getShape(shape, 0);
  assert(outputRank != -1 && "Shape tensor must have constant shape");

  // Initialize context and results.
  outputDims.resize(outputRank);

  // Shape values can be 0, -1, or N (N > 0).
  //   - 0: the output dim is setting to the input dim at the same index.
  //   Thus, it must happen at the index < dataRank.
  //   - -1: the output dim is calculated from the other output dims. No more
  //   than one dim in the output has value -1.

  // Compute the total number of elements using the input data operand.
  IndexExpr numOfElements = LiteralIndexExpr(1);
  for (unsigned i = 0; i < dataRank; ++i)
    numOfElements = numOfElements * createIE->getShapeAsDim(data, i);

  // Compute the total number of elements from the shape values.
  IndexExpr numOfElementsFromShape = LiteralIndexExpr(1);
  for (unsigned i = 0; i < outputRank; ++i) {
    IndexExpr dimShape = createIE->getIntFromArrayAsSymbol(shape, i);
    if (dimShape.isUndefined())
      return op->emitError("shape input parameter could not be processed");
    IndexExpr dim;
    if (i < dataRank)
      // dimShape == 0: use dim from the input.
      dim = dimShape.selectOrSelf(
          dimShape == 0, createIE->getShapeAsDim(data, i));
    else
      dim = dimShape;

    // Just store the dim as it is. Real value for -1 will be computed later.
    outputDims[i] = dim;

    // dimShape == -1: use 1 to compute the number of elements to avoid
    // negative value.
    dim = dim.selectOrSelf(dim == -1, LiteralIndexExpr(1));
    numOfElementsFromShape = numOfElementsFromShape * dim;
  }

  // All the output dims except the one with -1 are computed. Thus, only
  // update the dim with -1 here.
  for (unsigned i = 0; i < outputRank; ++i)
    outputDims[i] = outputDims[i].selectOrSelf(
        outputDims[i] == -1, numOfElements.floorDiv(numOfElementsFromShape));

  // Save the final result.
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

LogicalResult ONNXReshapeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape tensor is specified.
  if (!hasShapeAndRank(getData()) || !hasShapeAndRank(getShape()))
    return success();

  // Only rank 1 shape tensors are supported.
  auto shapeTensorTy = getShape().getType().cast<RankedTensorType>();
  if (shapeTensorTy.getShape().size() != 1)
    return emitError("Shape tensor must have rank one");

  // Shape tensor must have constant shape.
  int64_t outputRank = shapeTensorTy.getShape()[0];
  if (outputRank < 0)
    return emitError("Shape tensor must have constant shape");

  Type elementType = getData().getType().cast<ShapedType>().getElementType();
  ONNXReshapeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXReshapeOp>;
} // namespace onnx_mlir
