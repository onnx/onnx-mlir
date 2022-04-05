/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- ReverseSequence.cpp - Shape Inference for ReverseSequence Op ----===//
//
// This file implements shape inference for the ONNX ReverseSequence Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

ONNXReverseSequenceOpShapeHelper::ONNXReverseSequenceOpShapeHelper(
    ONNXReverseSequenceOp *newOp, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXReverseSequenceOp>(
          newOp, newOp->getOperation()->getNumResults(), inScope) {}

ONNXReverseSequenceOpShapeHelper::ONNXReverseSequenceOpShapeHelper(
    ONNXReverseSequenceOp *newOp, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXReverseSequenceOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal, inScope) {}

LogicalResult ONNXReverseSequenceOpShapeHelper::Compute(
    ONNXReverseSequenceOpAdaptor operandAdaptor) {

  // Get info about input data operand.
  Value input = operandAdaptor.input();
  MemRefBoundsIndexCapture inputBounds(input);
  int64_t inputRank = inputBounds.getRank();

  for (int64_t i = 0; i < inputRank; ++i)
    dimsForOutput(0).emplace_back(inputBounds.getDim(i));

  return success();
}

} // namespace onnx_mlir
