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
#include "src/Support/Diagnostic.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXConcatOpShapeHelper::computeShape(
    ONNXConcatOpAdaptor operandAdaptor) {
  unsigned numInputs = op->getNumOperands();
  Value firstInput = operandAdaptor.inputs().front();
  ArrayRef<int64_t> commonShape =
      firstInput.getType().cast<ShapedType>().getShape();
  int64_t commonRank = commonShape.size();
  int64_t axisIndex = op->axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(inputs).
  assert(-commonRank <= axisIndex && axisIndex < commonRank &&
         "axis out of range");

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

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir
