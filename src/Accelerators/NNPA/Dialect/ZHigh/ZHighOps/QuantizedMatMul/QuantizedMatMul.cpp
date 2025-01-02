/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- QuantizedMatMul.cpp - ZHigh Operations ----------------===//
//
// Copyright 2023 The IBM Research Authors.
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

LogicalResult ZHighQuantizedMatMulOpShapeHelper::computeShape() {
  ZHighQuantizedMatMulOp::Adaptor operandAdaptor(operands);
  // Output dims of result.
  DimsExpr outputDims;

  // Get operands.
  Value X = operandAdaptor.getX();
  Value Y = operandAdaptor.getY();

  // Get bounds
  SmallVector<IndexExpr, 4> XDims, YDims;
  createIE->getShapeAsDims(X, XDims);
  createIE->getShapeAsDims(Y, YDims);
  int64_t xRank = XDims.size();
  int64_t yRank = YDims.size();

  if (!(xRank == 2 || xRank == 3))
    return failure();

  if (xRank == 2) {
    // X :: MxN
    int64_t xI = 0;
    outputDims.emplace_back(XDims[xI]);
    // Y :: NxP
    int64_t yI = 1;
    outputDims.emplace_back(YDims[yI]);
  } else if (xRank == 3) {
    // X :: SxMxN
    outputDims.emplace_back(XDims[0]);
    outputDims.emplace_back(XDims[1]);
    if (yRank == 2) {
      // Y :: NxP
      outputDims.emplace_back(YDims[1]);
      isBroadcasted = true;
    } else if (yRank == 3) {
      // Y :: SxNxP
      outputDims.emplace_back(YDims[2]);
      isStacked = true;
    }
  }

  // Keep all original dimensions: M, N, P if 2D or S, M, N, P if 3D.
  if (xRank == 2) {
    // M
    allOriginalDims.emplace_back(XDims[0]);
    // N
    allOriginalDims.emplace_back(XDims[1]);
    // P
    allOriginalDims.emplace_back(YDims[1]);
  } else if (xRank == 3) {
    // S
    allOriginalDims.emplace_back(XDims[0]);
    // M
    allOriginalDims.emplace_back(XDims[1]);
    // N
    allOriginalDims.emplace_back(XDims[2]);
    // P
    if (yRank == 2)
      allOriginalDims.emplace_back(YDims[1]);
    else if (yRank == 3)
      allOriginalDims.emplace_back(YDims[2]);
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighQuantizedMatMulOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(getX()) || !hasRankedType(getY()))
    return success();

  bool dequantizeOutput = (getDequantizeOutput() == -1);
  ZHighQuantizedMatMulOpShapeHelper shapeHelper(getOperation());
  shapeHelper.computeShapeAndAssertOnFailure();

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.getOutputDims(), outputDims);
  Type elementType =
      mlir::cast<ShapedType>(getResult(0).getType()).getElementType();
  ZTensorEncodingAttr encoding;
  ZTensorEncodingAttr::QuantizedType qtype =
      ZTensorEncodingAttr::QuantizedType::DLFLOAT16;
  if (dequantizeOutput)
    qtype = ZTensorEncodingAttr::QuantizedType::UNDEFINED;
  if (outputDims.size() == 2)
    encoding = ZTensorEncodingAttr::get(
        this->getContext(), ZTensorEncodingAttr::DataLayout::_2D, qtype);
  else if (outputDims.size() == 3)
    encoding = ZTensorEncodingAttr::get(
        this->getContext(), ZTensorEncodingAttr::DataLayout::_3DS, qtype);

  updateType(getOperation(), getResult(0), outputDims, elementType, encoding);
  return success();
}

LogicalResult ZHighQuantizedMatMulOp::verify() {
  ZHighQuantizedMatMulOpAdaptor operandAdaptor(*this);
  // Get operands.
  Value X = operandAdaptor.getX();
  Value Y = operandAdaptor.getY();
  Value B = operandAdaptor.getB();

  if (!hasRankedType(X) || !hasRankedType(Y))
    return success();

  // Get layouts.
  ZTensorEncodingAttr::DataLayout xLayout = getZTensorLayout(X.getType());
  ZTensorEncodingAttr::DataLayout yLayout = getZTensorLayout(Y.getType());
  // Bias can be None.
  ZTensorEncodingAttr::DataLayout bLayout;
  bool hasBias = !mlir::isa<NoneType>(B.getType());
  if (hasBias) {
    if (!hasRankedType(B))
      return success();
    bLayout = getZTensorLayout(B.getType());
  }

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
