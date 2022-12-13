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
  // Output dims of result.
  DimsExpr outputDims;

  // Get operands and bounds.
  Value zGate = operandAdaptor.z_gate();
  MemRefBoundsIndexCapture zBounds(zGate);
  int64_t rank = zBounds.getRank();

  for (int64_t i = 0; i < rank - 1; ++i)
    outputDims.emplace_back(zBounds.getDim(i));
  IndexExpr lastDim = zBounds.getDim(rank - 1) * LiteralIndexExpr(3);
  outputDims.emplace_back(lastDim);

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighStickForGRUOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(z_gate()) && !hasRankedType(r_gate()) &&
      !hasRankedType(h_gate()))
    return success();

  Type elementType = getResult().getType().cast<ShapedType>().getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::ZRH);

  ZHighStickForGRUOpShapeHelper shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(elementType, encoding);
}

} // namespace zhigh
} // namespace onnx_mlir
