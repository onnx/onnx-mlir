/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ GRU.cpp - ZHigh Operations ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// ShapeHelper
//===----------------------------------------------------------------------===//

LogicalResult ZHighFixGRUYhOpShapeHelper::computeShape() {
  DimsExpr dims;
  createIE->getShapeAsDims(operands[0], dims);

  DimsExpr outputDims;
  outputDims.emplace_back(dims[1]);
  outputDims.emplace_back(dims[2]);
  outputDims.emplace_back(dims[3]);
  setOutputDims(outputDims);

  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighFixGRUYhOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasRankedType(getY()))
    return success();

  Type elementType = getElementType(getY().getType());
  ZHighFixGRUYhOpShapeHelper shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

} // namespace zhigh
} // namespace onnx_mlir
