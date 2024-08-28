/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ StickForGRU.cpp - ZHigh Operations ----------------===//
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

LogicalResult ZHighStickForGRUOpShapeHelper::computeShape() {
  ZHighStickForGRUOp::Adaptor operandAdaptor(operands);
  Value zGate = operandAdaptor.getZGate();

  // Output dims of result.
  DimsExpr outputDims;

  // Get operands and bounds.
  SmallVector<IndexExpr, 4> zGateDims;
  createIE->getShapeAsDims(zGate, zGateDims);
  int64_t rank = zGateDims.size();

  for (int64_t i = 0; i < rank - 1; ++i)
    outputDims.emplace_back(zGateDims[i]);
  IndexExpr lastDim = zGateDims[rank - 1] * LitIE(3);
  outputDims.emplace_back(lastDim);

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighStickForGRUOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasRankedType(getZGate()) && !hasRankedType(getRGate()) &&
      !hasRankedType(getHGate()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getResult().getType()).getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::ZRH);

  ZHighStickForGRUOpShapeHelper shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(elementType, encoding);
}

} // namespace zhigh
} // namespace onnx_mlir
