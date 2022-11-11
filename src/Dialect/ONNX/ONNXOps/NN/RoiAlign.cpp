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

ONNXRoiAlignOpShapeHelper::ONNXRoiAlignOpShapeHelper(
    ONNXRoiAlignOp *newOp, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXRoiAlignOp>(
          newOp, newOp->getOperation()->getNumResults(), inScope),
      xDims(), batchIndicesDims() {}

ONNXRoiAlignOpShapeHelper::ONNXRoiAlignOpShapeHelper(ONNXRoiAlignOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXRoiAlignOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal, inScope),
      xDims(), batchIndicesDims() {}

LogicalResult ONNXRoiAlignOpShapeHelper::computeShape(
    ONNXRoiAlignOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // get input info.
  Value X = operandAdaptor.X();
  Value batch_indices = operandAdaptor.batch_indices();

  // Read X and batch_indices shapes as dim indices.
  MemRefBoundsIndexCapture xBounds(X);
  MemRefBoundsIndexCapture batchIndicesBounds(batch_indices);
  xBounds.getDimList(xDims);
  batchIndicesBounds.getDimList(batchIndicesDims);

  int64_t height = op->output_height();
  int64_t width = op->output_width();

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
  mlir::Value X = operandAdaptor.X();
  mlir::Value batch_indices = operandAdaptor.batch_indices();

  if (!hasShapeAndRank(X) || !hasShapeAndRank(batch_indices))
    return success();

  int64_t x_rank = X.getType().cast<mlir::ShapedType>().getRank();
  int64_t batch_indices_rank =
      batch_indices.getType().cast<mlir::ShapedType>().getRank();

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
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!X().getType().isa<RankedTensorType>() ||
      !batch_indices().getType().isa<RankedTensorType>())
    return success();

  auto elementType = X().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXRoiAlignOpShapeHelper, ONNXRoiAlignOp,
      ONNXRoiAlignOpAdaptor>(*this, elementType);
}
