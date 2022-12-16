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

LogicalResult ZHighStickForGRUOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(z_gate()) && !hasRankedType(r_gate()) &&
      !hasRankedType(h_gate()))
    return success();

  ZHighStickForGRUOpAdaptor operandAdaptor(*this);
  ZHighStickForGRUOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError(
        "Failed to scan ZHigh StickForGRU parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  Type elementType = getResult().getType().cast<ShapedType>().getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::ZRH);
  updateType(getResult(), outputDims, elementType, encoding);
  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
