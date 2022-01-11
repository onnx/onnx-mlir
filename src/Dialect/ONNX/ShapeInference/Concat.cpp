/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Concat.cpp - Shape Inference for Concat Op ----------------===//
//
// This file implements shape inference for the ONNX Concat Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

ONNXConcatOpShapeHelper::ONNXConcatOpShapeHelper(ONNXConcatOp *newOp)
    : ONNXOpShapeHelper<ONNXConcatOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXConcatOpShapeHelper::ONNXConcatOpShapeHelper(ONNXConcatOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXConcatOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXConcatOpShapeHelper::computeShape(
    ONNXConcatOpAdaptor operandAdaptor) {

  int inputNum = op->getNumOperands();
  Value firstInput = operandAdaptor.getODSOperands(0)[0];
  auto commonType = firstInput.getType().cast<ShapedType>();
  auto commonShape = commonType.getShape();
  auto commonRank = commonShape.size();
  int64_t axisIndex = op->axis();

  // Negative axis means values are counted from the opposite side.
  // TOFIX should be in normalization pass
  if (axisIndex < 0) {
    axisIndex = commonRank + axisIndex;
  }

  IndexExpr cumulativeAxisSize = LiteralIndexExpr(0);
  for (int i = 0; i < inputNum; ++i) {
    Value currentInput = operandAdaptor.getODSOperands(0)[i];
    MemRefBoundsIndexCapture currInputBounds(currentInput);
    DimIndexExpr currentSize(currInputBounds.getDim(axisIndex));
    cumulativeAxisSize = cumulativeAxisSize + currentSize;
  }

  DimsExpr outputDims;
  MemRefBoundsIndexCapture firstInputBounds(firstInput);
  outputDims.resize(commonRank);
  for (unsigned int i = 0; i < commonRank; i++) {
    if (i == axisIndex) {
      outputDims[i] = cumulativeAxisSize;
    } else {
      outputDims[i] = firstInputBounds.getDim(i);
    }
  }

  dimsForOutput(0) = outputDims;
  return success();
}
