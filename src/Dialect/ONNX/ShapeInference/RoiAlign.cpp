/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- RoiAlign.cpp - Shape Inference for RoiAlign Op
//--------------===//
//
// This file implements shape inference for the ONNX Shape Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

ONNXRoiAlignOpShapeHelper::ONNXRoiAlignOpShapeHelper(ONNXRoiAlignOp *newOp)
    : ONNXOpShapeHelper<ONNXRoiAlignOp>(
          newOp, newOp->getOperation()->getNumResults()),
      xDims(), batchIndicesDims() {}

ONNXRoiAlignOpShapeHelper::ONNXRoiAlignOpShapeHelper(ONNXRoiAlignOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXRoiAlignOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal),
      xDims(), batchIndicesDims() {}

LogicalResult ONNXRoiAlignOpShapeHelper::computeShape(
    ONNXRoiAlignOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // get input info.
  Value X = operandAdaptor.X();
  Value batch_indices = operandAdaptor.batch_indices();

  // Test ranks.
  if (X.getType().cast<ShapedType>().getShape().size() != 4)
    return op->emitError("RoiAlign with X should be a 4D tensor");
  if (batch_indices.getType().cast<ShapedType>().getShape().size() != 1)
    return op->emitError("RoiAlign with batch_indices should be a 1D tensor");

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
  dimsForOutput(0) = outputDims;

  return success();
}
