/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ LSTM.cpp - ZHigh Operations -----------------------===//
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

LogicalResult ZHighLSTMOp::verify() {
  ZHighLSTMOpAdaptor operandAdaptor(*this);
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
    if (!ShapedType::isDynamic(dim2) && (dim2 != hiddenSize * 4))
      return failure();
  }

  // Verify hidden size in R.
  if (hasRankedType(R)) {
    int64_t dim1 = R.getType().cast<RankedTensorType>().getShape()[1];
    int64_t dim2 = R.getType().cast<RankedTensorType>().getShape()[2];
    if (!ShapedType::isDynamic(dim1) && (dim1 != hiddenSize))
      return failure();
    if (!ShapedType::isDynamic(dim2) && (dim2 != hiddenSize * 4))
      return failure();
  }

  // Verify hidden size in WB.
  if (!WB.getType().isa<NoneType>() && hasRankedType(WB)) {
    int64_t dim1 = WB.getType().cast<RankedTensorType>().getShape()[1];
    if (!ShapedType::isDynamic(dim1) && (dim1 != hiddenSize * 4))
      return failure();
  }

  // Verify hidden size in RB.
  if (!RB.getType().isa<NoneType>() && hasRankedType(RB)) {
    int64_t dim1 = RB.getType().cast<RankedTensorType>().getShape()[1];
    if (!ShapedType::isDynamic(dim1) && (dim1 != hiddenSize * 4))
      return failure();
  }

  return success();
}

LogicalResult ZHighLSTMOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()) || !hasRankedType(hidden_weights()))
    return success();

  Builder builder(getContext());
  ZHighLSTMOpAdaptor operandAdaptor(*this);
  ZHighLSTMOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh LSTM parameters successfully");

  // Output type is 4DS.
  SmallVector<int64_t, 4> hnOutputDims, cfOutputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), hnOutputDims);
  IndexExpr::getShape(shapeHelper.dimsForOutput(1), cfOutputDims);
  Type elementType = input().getType().cast<ShapedType>().getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::_4DS);
  updateType(getResults()[0], hnOutputDims, elementType, encoding);
  updateType(getResults()[1], cfOutputDims, elementType, encoding);
  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
