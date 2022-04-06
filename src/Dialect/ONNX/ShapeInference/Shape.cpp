/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Shape.cpp - Shape Inference for Shape Op --------------===//
//
// This file implements shape inference for the ONNX Shape Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

ONNXShapeOpShapeHelper::ONNXShapeOpShapeHelper(
    ONNXShapeOp *newOp, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXShapeOp>(
          newOp, newOp->getOperation()->getNumResults(), inScope) {}

ONNXShapeOpShapeHelper::ONNXShapeOpShapeHelper(ONNXShapeOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXShapeOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal, inScope) {}

LogicalResult ONNXShapeOpShapeHelper::computeShape(
    ONNXShapeOpAdaptor operandAdaptor) {

  // Get info about input data operand.
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t dataRank = dataBounds.getRank();

  // To be initalized from op (opset > 13)
  int64_t start = 0;
  int64_t end = dataRank; // Default value if option not defined.

  // Handle negative values.
  if (start < 0)
    start = start + dataRank;
  if (end < 0)
    end = end + dataRank;
  if (start < 0 || start > dataRank)
    return op->emitError("start value is out of bound");
  if (end < 0 || end > dataRank)
    return op->emitError("end value is out of bound");

  // Save actual values in selected data
  for (int64_t i = start; i < end; ++i)
    selectedData.emplace_back(dataBounds.getDim(i));
  // Output is the actual number of values (1D)
  dimsForOutput(0).emplace_back(LiteralIndexExpr(selectedData.size()));
  return success();
}

} // namespace onnx_mlir
