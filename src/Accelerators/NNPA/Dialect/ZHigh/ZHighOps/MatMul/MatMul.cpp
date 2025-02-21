/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ MatMul.cpp - ZHigh Operations ---------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
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

LogicalResult ZHighMatMulOpShapeHelper::computeShape() {
  ZHighMatMulOp matmulOp = llvm::dyn_cast<ZHighMatMulOp>(op);
  ZHighMatMulOp::Adaptor operandAdaptor(operands);
  // Output dims of result.
  DimsExpr outputDims;

  // Get operands.
  Value X = operandAdaptor.getX();
  Value Y = operandAdaptor.getY();

  // Get transpose attributes.
  int64_t transposeA = (matmulOp.getTransposeA() != 0) ? 1 : 0;
  int64_t transposeB = (matmulOp.getTransposeB() != 0) ? 1 : 0;

  // Get bounds
  SmallVector<IndexExpr, 4> XDims, YDims;
  createIE->getShapeAsDims(X, XDims);
  createIE->getShapeAsDims(Y, YDims);
  int64_t xRank = XDims.size();
  int64_t yRank = YDims.size();

  if (!(xRank == 2 || xRank == 3))
    return failure();

  // Determine the dimensions of the output tensor.
  if (xRank == 2) {
    // X :: MxN
    int64_t xI = 0;
    if (transposeA)
      // X :: NxM
      xI = 1;
    if (yRank == 2) {
      // Y :: NxP
      int64_t yI = 1;
      if (transposeB)
        // Y :: PxN
        yI = 0;
      // Unstacked case: X:2D (m,n) - Y:2D (n,p) - Bias:1D (p) - Out:2D (m,p)
      outputDims.emplace_back(XDims[xI]);
      outputDims.emplace_back(YDims[yI]);
    } else if (yRank == 3) {
      // Y :: SxNxP
      int64_t yI1 = 0;
      int64_t yI2 = 2;
      if (transposeB) {
        // Y :: SxPxN
        yI2 = 1;
      }
      // Broadcast 1 case: X:2D (m,n) - Y:3DS (s,n,p) - Bias:2DS (s,p) - Out:3DS
      // (s,m,p)
      outputDims.emplace_back(YDims[yI1]);
      outputDims.emplace_back(XDims[xI]);
      outputDims.emplace_back(YDims[yI2]);
      isBroadcasted1 = true;
    }
  } else if (xRank == 3) {
    // X :: SxMxN
    int64_t xI1 = 0;
    int64_t xI2 = 1;
    if (transposeA)
      // X :: SxNxM
      xI2 = 2;
    if (yRank == 2) {
      // Y :: NxP
      int64_t yI = 1;
      if (transposeB)
        // Y :: PxN
        yI = 0;
      // Broadcast 23 case: X:3DS (s,m,n) - Y:2D (n,p) - Bias:1D (p) - Out:3DS
      // (s,m,p)
      outputDims.emplace_back(XDims[xI1]);
      outputDims.emplace_back(XDims[xI2]);
      outputDims.emplace_back(YDims[yI]);
      isBroadcasted23 = true;
    } else if (yRank == 3) {
      // Y :: SxNxP
      int64_t yI = 2;
      if (transposeB)
        // Y :: SxPxN
        yI = 1;
      // Stacked case: X:3DS (s,m,n) - Y:3DS (s,n,p) - Bias:2DS (s,p) - Out:3DS
      // (s,m,p)
      outputDims.emplace_back(XDims[xI1]);
      outputDims.emplace_back(XDims[xI2]);
      outputDims.emplace_back(YDims[yI]);
      isStacked = true;
    }
  }

  // Keep all original dimensions: M, N, P if 2D or S, M, N, P if 3D.
  if (xRank == 2) {
    if (transposeA) {
      // M
      allOriginalDims.emplace_back(XDims[1]);
      // N
      allOriginalDims.emplace_back(XDims[0]);
    } else {
      // M
      allOriginalDims.emplace_back(XDims[0]);
      // N
      allOriginalDims.emplace_back(XDims[1]);
    }
    if (yRank == 2) {
      // P
      if (transposeB)
        allOriginalDims.emplace_back(YDims[0]);
      else
        allOriginalDims.emplace_back(YDims[1]);
    } else if (yRank == 3) {
      // S
      allOriginalDims.emplace_back(YDims[0]);
      // P
      if (transposeB)
        allOriginalDims.emplace_back(YDims[1]);
      else
        allOriginalDims.emplace_back(YDims[2]);
    }
  } else if (xRank == 3) {
    // S
    allOriginalDims.emplace_back(XDims[0]);
    if (transposeA) {
      // M
      allOriginalDims.emplace_back(XDims[2]);
      // N
      allOriginalDims.emplace_back(XDims[1]);
    } else {
      // M
      allOriginalDims.emplace_back(XDims[1]);
      // N
      allOriginalDims.emplace_back(XDims[2]);
    }
    // P
    if (yRank == 2)
      if (transposeB)
        allOriginalDims.emplace_back(YDims[0]);
      else
        allOriginalDims.emplace_back(YDims[1]);
    else if (yRank == 3) {
      if (transposeB)
        allOriginalDims.emplace_back(YDims[1]);
      else
        allOriginalDims.emplace_back(YDims[2]);
    }
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighMatMulOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasRankedType(getX()) || !hasRankedType(getY()))
    return success();

  ZHighMatMulOpShapeHelper shapeHelper(getOperation());
  shapeHelper.computeShapeAndAssertOnFailure();

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.getOutputDims(), outputDims);
  Type elementType =
      mlir::cast<ShapedType>(getResult().getType()).getElementType();
  ZTensorEncodingAttr encoding;
  if (outputDims.size() == 2)
    encoding = ZTensorEncodingAttr::get(
        this->getContext(), ZTensorEncodingAttr::DataLayout::_2D);
  else if (outputDims.size() == 3)
    encoding = ZTensorEncodingAttr::get(
        this->getContext(), ZTensorEncodingAttr::DataLayout::_3DS);

  updateType(getOperation(), getResult(), outputDims, elementType, encoding);
  return success();
}

LogicalResult ZHighMatMulOp::verify() {
  ZHighMatMulOpAdaptor operandAdaptor(*this);
  // Get operands.
  Value X = operandAdaptor.getX();
  Value Y = operandAdaptor.getY();
  Value B = operandAdaptor.getB();

  // Get layouts.
  ZTensorEncodingAttr::DataLayout xLayout = getZTensorLayout(X.getType());
  ZTensorEncodingAttr::DataLayout yLayout = getZTensorLayout(Y.getType());
  // Bias can be None.
  ZTensorEncodingAttr::DataLayout bLayout;
  bool hasBias = !mlir::isa<NoneType>(B.getType());
  if (hasBias)
    bLayout = getZTensorLayout(B.getType());

  // X must be 2D or 3DS.
  if (!((xLayout == ZTensorEncodingAttr::DataLayout::_2D) ||
          (xLayout == ZTensorEncodingAttr::DataLayout::_3DS)))
    return failure();

  // If X is 2D, Y must be 2D or 3DS.
  // If X is 2D and Y is 2D, B must be 1D.
  // If X is 2D and Y is 3DS, B must be 2DS.
  if (xLayout == ZTensorEncodingAttr::DataLayout::_2D) {
    if (!((yLayout == ZTensorEncodingAttr::DataLayout::_2D) ||
            (yLayout == ZTensorEncodingAttr::DataLayout::_3DS)))
      return failure();
    if (yLayout == ZTensorEncodingAttr::DataLayout::_2D) {
      if (hasBias && !(bLayout == ZTensorEncodingAttr::DataLayout::_1D))
        return failure();
    } else if (yLayout == ZTensorEncodingAttr::DataLayout::_3DS) {
      if (hasBias && !(bLayout == ZTensorEncodingAttr::DataLayout::_2DS))
        return failure();
    }
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
