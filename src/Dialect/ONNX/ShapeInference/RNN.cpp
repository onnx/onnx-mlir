/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- RNN.cpp - Shape Inference for RNN Op -----------------===//
//
// This file implements shape inference for the ONNX RNN Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

ONNXLRNOpShapeHelper::ONNXLRNOpShapeHelper(ONNXLRNOp *newOp)
    : ONNXOpShapeHelper<ONNXLRNOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXLRNOpShapeHelper::ONNXLRNOpShapeHelper(ONNXLRNOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXLRNOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXLRNOpShapeHelper::computeShape(
    ONNXLRNOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Basic information.
  auto rank = operandAdaptor.X().getType().cast<ShapedType>().getRank();

  // Perform transposition according to perm attribute.
  DimsExpr outputDims;
  MemRefBoundsIndexCapture XBounds(operandAdaptor.X());
  for (decltype(rank) i = 0; i < rank; ++i) {
    outputDims.emplace_back(XBounds.getDim(i));
  }

  // Set type for the first output.
  dimsForOutput(0) = outputDims;
  return success();
}
