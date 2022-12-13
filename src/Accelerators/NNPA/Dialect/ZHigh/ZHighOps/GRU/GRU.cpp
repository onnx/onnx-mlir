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

LogicalResult ZHighGRUOp::verify() {
  ZHighGRUOpAdaptor operandAdaptor(*this);
  // Get operands.
  Value W = operandAdaptor.input_weights();
  Value R = operandAdaptor.hidden_weights();
  Value WB = operandAdaptor.input_bias();
  Value RB = operandAdaptor.hidden_bias();

  // Hidden size attribute.
  int64_t hiddenSize = hidden_size();

  // Verify hidden size in W.
  if (hasRankedType(W)) {
    int64_t dim2 = W.getType().cast<RankedTensorType>().getShape()[2];
    if (!ShapedType::isDynamic(dim2) && (dim2 != hiddenSize * 3))
      return failure();
  }

  // Verify hidden size in R.
  if (hasRankedType(R)) {
    int64_t dim1 = R.getType().cast<RankedTensorType>().getShape()[1];
    int64_t dim2 = R.getType().cast<RankedTensorType>().getShape()[2];
    if (!ShapedType::isDynamic(dim1) && (dim1 != hiddenSize))
      return failure();
    if (!ShapedType::isDynamic(dim2) && (dim2 != hiddenSize * 3))
      return failure();
  }

  // Verify hidden size in WB.
  if (!WB.getType().isa<NoneType>() && hasRankedType(WB)) {
    int64_t dim1 = WB.getType().cast<RankedTensorType>().getShape()[1];
    if (!ShapedType::isDynamic(dim1) && (dim1 != hiddenSize * 3))
      return failure();
  }

  // Verify hidden size in RB.
  if (!RB.getType().isa<NoneType>() && hasRankedType(RB)) {
    int64_t dim1 = RB.getType().cast<RankedTensorType>().getShape()[1];
    if (!ShapedType::isDynamic(dim1) && (dim1 != hiddenSize * 3))
      return failure();
  }

  return success();
}

LogicalResult ZHighGRUOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()) || !hasRankedType(hidden_weights()))
    return success();

  Builder builder(getContext());
  ZHighGRUOpAdaptor operandAdaptor(*this);
  ZHighGRUOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh GRU parameters successfully");

  SmallVector<int64_t, 4> hnOutputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), hnOutputDims);
  Type elementType = input().getType().cast<ShapedType>().getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::_4DS);
  updateType(getResult(), hnOutputDims, elementType, encoding);
  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
