/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ MatMul.cpp - ZHigh Operations ---------------------===//
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

LogicalResult ZHighMatMulOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()) || !hasRankedType(Y()))
    return success();

  ZHighMatMulOpAdaptor operandAdaptor(*this);
  ZHighMatMulOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh MatMul parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  Type elementType = getResult().getType().cast<ShapedType>().getElementType();
  ZTensorEncodingAttr encoding;
  if (outputDims.size() == 2)
    encoding = ZTensorEncodingAttr::get(
        this->getContext(), ZTensorEncodingAttr::DataLayout::_2D);
  else if (outputDims.size() == 3)
    encoding = ZTensorEncodingAttr::get(
        this->getContext(), ZTensorEncodingAttr::DataLayout::_3DS);

  updateType(getResult(), outputDims, elementType, encoding);
  return success();
}

LogicalResult ZHighMatMulOp::verify() {
  ZHighMatMulOpAdaptor operandAdaptor(*this);
  // Get operands.
  Value X = operandAdaptor.X();
  Value Y = operandAdaptor.Y();
  Value B = operandAdaptor.B();

  // Get layouts.
  ZTensorEncodingAttr::DataLayout xLayout = getZTensorLayout(X.getType());
  ZTensorEncodingAttr::DataLayout yLayout = getZTensorLayout(Y.getType());
  // Bias can be None.
  ZTensorEncodingAttr::DataLayout bLayout;
  bool hasBias = !B.getType().isa<NoneType>();
  if (hasBias)
    bLayout = getZTensorLayout(B.getType());

  // X must be 2D or 3DS.
  if (!((xLayout == ZTensorEncodingAttr::DataLayout::_2D) ||
          (xLayout == ZTensorEncodingAttr::DataLayout::_3DS)))
    return failure();

  // If X is 2D, Y must be 2D and B must be 1D
  if (xLayout == ZTensorEncodingAttr::DataLayout::_2D) {
    if (!(yLayout == ZTensorEncodingAttr::DataLayout::_2D))
      return failure();
    if (hasBias && !(bLayout == ZTensorEncodingAttr::DataLayout::_1D))
      return failure();
  }

  // X is 3DS, valid types for (X, Y, B) are (3DS, 3DS, 2DS) or (3DS, 2D, 1D)
  if (xLayout == ZTensorEncodingAttr::DataLayout::_3DS) {
    if (yLayout == ZTensorEncodingAttr::DataLayout::_3DS) {
      if (hasBias && !(bLayout == ZTensorEncodingAttr::DataLayout::_2DS))
        return failure();
    } else if (yLayout == ZTensorEncodingAttr::DataLayout::_2D) {
      if (hasBias && !(bLayout == ZTensorEncodingAttr::DataLayout::_1D))
        return failure();
    }
  }

  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
