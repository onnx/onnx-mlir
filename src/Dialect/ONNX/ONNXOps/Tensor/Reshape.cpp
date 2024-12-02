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
  int64_t dataRank = mlir::cast<ShapedType>(data.getType()).getShape().size();

  // Get info about shape operand.
  Value shape = operandAdaptor.getShape();
  int64_t outputRank = createIE->getShape(shape, 0);
  assert(outputRank != ShapedType::kDynamic &&
         "Shape tensor must have constant shape");

  // Initialize context and results.
  outputDims.resize(outputRank);

  // Shape values can be 0, -1, or N (N > 0).
  //   - 0: the output dim is setting to the input dim at the same index.
  //   Thus, it must happen at the index < dataRank.
  //   - -1: the output dim is calculated from the other output dims. No more
  //   than one dim in the output has value -1.

  // Compute the total number of elements using the input data operand.
  // dataRank will be 0 if Data is unranked tensor.
  // The number of element will not be computed
  IndexExpr numOfElements = LitIE(1);
  for (unsigned i = 0; i < dataRank; ++i)
    numOfElements = numOfElements * createIE->getShapeAsDim(data, i);

  // Compute the total number of elements from the shape values.
  IndexExpr numOfElementsFromShape = LitIE(1);
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
    dim = dim.selectOrSelf(dim == -1, LitIE(1));
    numOfElementsFromShape = numOfElementsFromShape * dim;
  }

  // When data is ranked tensor, all the output dims except the one with -1
  // are computed. Thus, only update the dim with -1 here.
  // When data is unranked tensor, output dims with -1 or 0 (allowzero == 0)
  // should be -1 (represented as QuestionmarkIndexExpr)
  for (unsigned i = 0; i < outputRank; ++i) {
    if (hasShapeAndRank(data)) {
      IndexExpr dimShape = createIE->getIntFromArrayAsSymbol(shape, i);
      outputDims[i] = outputDims[i].selectOrSelf(
          dimShape == -1, numOfElements.floorDiv(numOfElementsFromShape));
    } else {
      // ToFix: can not check getAllowzero because the operandAdaptor is
      // constructed without attributes
      // Anyway the question mark is a conservative but correct result.
      outputDims[i] = outputDims[i].selectOrSelf(
          outputDims[i] == 0, QuestionmarkIndexExpr(false));
      outputDims[i] = outputDims[i].selectOrSelf(
          outputDims[i] == -1, QuestionmarkIndexExpr(false));
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

LogicalResult ONNXReshapeOp::verify() {
  // Cannot verify if shape has unknown rank.
  if (!hasShapeAndRank(getShape()))
    return success();

  // Only rank 1 shape tensors are supported.
  auto shapeTy = cast<ShapedType>(getShape().getType());
  if (shapeTy.getRank() != 1)
    return emitOpError("Shape tensor must have rank one");

  // TODO: Check that any -1 dim is used correctly.
  // TODO: Check that any 0 dim is used correctly with allowzero.
  // TODO: Check that data can reshape to shape if data's shape is known.

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXReshapeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape without data rank and static shape of shape.
  // If shape is constant shape, the rank of the output known.
  // This step may be helpful to reach the fix point.
  // TODO: Infer shape without data rank if shape is a constant
  //       without -1 and without 0 and allowzero.
  if (!hasShapeAndRank(getData()) && !hasStaticShape(getShape().getType()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getData().getType()).getElementType();
  ONNXReshapeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXReshapeOp>;
} // namespace onnx_mlir
