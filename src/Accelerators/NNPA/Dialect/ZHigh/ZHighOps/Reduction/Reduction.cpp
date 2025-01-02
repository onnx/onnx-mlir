/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Reduction.cpp - ZHigh Operations ----------------===//
//
// Copyright 2024 The IBM Research Authors.
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
template <typename OP_TYPE>
LogicalResult ZHighReductionOpShapeHelper<OP_TYPE>::computeShape() {
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());

  // Get operand.
  Value data = operandAdaptor.getData();

  // Get Rank
  int64_t rank = createIE->getShapedTypeRank(data);

  // Output dims of result.
  DimsExpr outputDims;

  // Get operands and bounds.
  SmallVector<IndexExpr, 4> inputDims;
  createIE->getShapeAsDims(data, inputDims);

  int64_t axis = rank - 1;
  LiteralIndexExpr one(1);
  // Copy the input until the second to last dimension
  for (int64_t i = 0; i < axis; ++i) {
    outputDims.emplace_back(inputDims[i]);
  }
  // The innermost dimension or last dimension needs to be reduced to one
  outputDims.emplace_back(
      one); // NNPA is always true for keepdims so we will reduce the dimension

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// ZHigh Shape Helper template instantiation
// Keep template instantiation at the end of the file.
//===----------------------------------------------------------------------===//

template struct ZHighReductionOpShapeHelper<ZHighReduceMaxOp>;
template struct ZHighReductionOpShapeHelper<ZHighReduceMinOp>;

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//
template <class OP_TYPE>
static LogicalResult inferShapeForReductionOps(OP_TYPE &op) {
  typename OP_TYPE::Adaptor operandAdaptor(op);
  if (!hasRankedType(operandAdaptor.getData()))
    return success();
  RankedTensorType dataType =
      mlir::cast<RankedTensorType>(operandAdaptor.getData().getType());
  ZHighReductionOpShapeHelper<OP_TYPE> shapeHelper(op.getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      dataType.getElementType(), dataType.getEncoding());
}

//===----------------------------------------------------------------------===//
// ReduceMax
//===----------------------------------------------------------------------===//

LogicalResult ZHighReduceMaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ZHighReduceMaxOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMin
//===----------------------------------------------------------------------===//

LogicalResult ZHighReduceMinOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ZHighReduceMinOp>(*this);
}

} // namespace zhigh
} // namespace onnx_mlir
