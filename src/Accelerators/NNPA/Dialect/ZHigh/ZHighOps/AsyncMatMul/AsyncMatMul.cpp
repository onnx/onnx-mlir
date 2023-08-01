/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- AsyncMatMul.cpp - ZHigh Operations-------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
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
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighMatMulAsyncOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(getX()) || !hasRankedType(getY()))
    return success();

  ZHighMatMulOpShapeHelper shapeHelper(getOperation());
  shapeHelper.computeShapeAndAssertOnFailure();

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.getOutputDims(), outputDims);
  Type elementType =
      getResults()[0].getType().cast<ShapedType>().getElementType();

  ZTensorEncodingAttr encoding;
  if (outputDims.size() == 2)
    encoding = ZTensorEncodingAttr::get(
        this->getContext(), ZTensorEncodingAttr::DataLayout::_2D);
  else if (outputDims.size() == 3)
    encoding = ZTensorEncodingAttr::get(
        this->getContext(), ZTensorEncodingAttr::DataLayout::_3DS);

  updateType(getResults()[0], outputDims, elementType, encoding);
  return success();
}
} // namespace zhigh
} // namespace onnx_mlir
