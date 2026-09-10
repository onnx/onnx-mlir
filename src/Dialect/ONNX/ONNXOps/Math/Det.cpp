/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- Det.cpp - ONNX Operations ------------------------===//
//
// This file provides definition of ONNX dialect Det operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

LogicalResult ONNXDetOpShapeHelper::computeShape() {
  ONNXDetOpAdaptor operandAdaptor(operands);
  Value X = operandAdaptor.getX();

  int64_t rank = createIE->getShapedTypeRank(X);
  if (rank < 2)
    return op->emitError("Det: input tensor must have rank >= 2");

  DimsExpr xDims;
  createIE->getShapeAsDims(X, xDims);

  // The innermost two dimensions must form a square matrix. Only checked
  // when both are statically known; a mismatch between dynamic dims is
  // instead a runtime error, same as elsewhere in the codebase.
  IndexExpr rowDim = xDims[rank - 2];
  IndexExpr colDim = xDims[rank - 1];
  if (rowDim.isLiteral() && colDim.isLiteral() &&
      rowDim.getLiteral() != colDim.getLiteral())
    return op->emitError(
        "Det: the innermost two dimensions must form a square matrix");

  // Output shape is the input shape without its last two (square matrix)
  // dimensions.
  DimsExpr outputDims(xDims.begin(), xDims.begin() + (rank - 2));

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// ONNXDetOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXDetOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()))
    return success();

  Type elementType = mlir::cast<ShapedType>(getX().getType()).getElementType();
  ONNXDetOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
