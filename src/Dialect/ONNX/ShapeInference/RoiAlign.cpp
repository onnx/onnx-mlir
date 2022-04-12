/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- RoiAlign.cpp - Shape Inference for RoiAlign Op --------===//
//
// This file implements shape inference for the ONNX RoiAlign Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

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
  dimsForOutput() = outputDims;

  return success();
}

} // namespace onnx_mlir
