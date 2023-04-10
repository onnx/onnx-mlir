/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ RoiAlign.cpp - ONNX Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect  operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

LogicalResult ONNXRoiAlignOpShapeHelper::computeShape() {
  // Get input info.
  ONNXRoiAlignOp roiAlignOp = llvm::cast<ONNXRoiAlignOp>(op);
  ONNXRoiAlignOpAdaptor operandAdaptor(operands);

  Value X = operandAdaptor.getX();
  Value batch_indices = operandAdaptor.getBatchIndices();

  // Read X and batch_indices shapes as dim indices.
  createIE->getShapeAsDims(X, xDims);
  createIE->getShapeAsDims(batch_indices, batchIndicesDims);

  int64_t height = roiAlignOp.getOutputHeight();
  int64_t width = roiAlignOp.getOutputWidth();

  DimsExpr outputDims = {batchIndicesDims[0], xDims[1],
      LiteralIndexExpr(height), LiteralIndexExpr(width)};

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXRoiAlignOp::verify() {
  ONNXRoiAlignOpAdaptor operandAdaptor = ONNXRoiAlignOpAdaptor(*this);
  // get input info.
  Value X = operandAdaptor.getX();
  Value batch_indices = operandAdaptor.getBatchIndices();

  if (!hasShapeAndRank(X) || !hasShapeAndRank(batch_indices))
    return success();

  int64_t x_rank = X.getType().cast<ShapedType>().getRank();
  int64_t batch_indices_rank =
      batch_indices.getType().cast<ShapedType>().getRank();

  // Test ranks.
  if (x_rank != 4)
    return emitOpError("RoiAlign with X should be a 4D tensor");
  if (batch_indices_rank != 1)
    return emitOpError("RoiAlign with batch_indices should be a 1D tensor");

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRoiAlignOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getBatchIndices()))
    return success();

  Type elementType = getX().getType().cast<ShapedType>().getElementType();
  ONNXRoiAlignOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
