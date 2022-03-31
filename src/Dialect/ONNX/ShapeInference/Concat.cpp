/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Concat.cpp - Shape Inference for Concat Op ----------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
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

  unsigned numInputs = op->getNumOperands();
  Value firstInput = operandAdaptor.inputs().front();
  ArrayRef<int64_t> commonShape =
      firstInput.getType().cast<ShapedType>().getShape();
  size_t commonRank = commonShape.size();
  int64_t axisIndex = op->axis();

  // Negative axis means values are counted from the opposite side.
  // TOFIX should be in normalization pass
  if (axisIndex < 0)
    axisIndex += commonRank;

  IndexExpr cumulativeAxisSize = LiteralIndexExpr(0);
  for (unsigned i = 0; i < numInputs; ++i) {
    Value currentInput = operandAdaptor.inputs()[i];
    MemRefBoundsIndexCapture currInputBounds(currentInput);
    DimIndexExpr currentSize(currInputBounds.getDim(axisIndex));
    cumulativeAxisSize = cumulativeAxisSize + currentSize;
  }

  DimsExpr outputDims(commonRank);
  MemRefBoundsIndexCapture firstInputBounds(firstInput);
  for (unsigned i = 0; i < commonRank; i++)
    outputDims[i] =
        (i == axisIndex) ? cumulativeAxisSize : firstInputBounds.getDim(i);

  dimsForOutput(0) = outputDims;
  return success();
}
