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

  // For Concat Op, the size of each dimension of inputs should be the same,
  // except for concatenated dimension. To simplify the result, constant
  // size is used if there is one. Otherwise, the dimension of the first
  // input tensor (implementation dependent) is used for the output tensor.
  DimsExpr outputDims(commonRank);
  MemRefBoundsIndexCapture firstInputBounds(operandAdaptor.inputs()[0]);
  for (unsigned dim = 0; dim < commonRank; dim++) {
    outputDims[dim] = firstInputBounds.getDim(dim);
  }
  IndexExpr cumulativeAxisSize =
      DimIndexExpr(firstInputBounds.getDim(axisIndex));

  // Handle the rest of input
  for (unsigned i = 1; i < numInputs; ++i) {
    Value currentInput = operandAdaptor.inputs()[i];
    MemRefBoundsIndexCapture currInputBounds(currentInput);
    for (unsigned dim = 0; dim < commonRank; dim++) {
      if (dim == axisIndex) {
        DimIndexExpr currentSize(currInputBounds.getDim(axisIndex));
        cumulativeAxisSize = cumulativeAxisSize + currentSize;
      } else {
        if (currInputBounds.getDim(dim).isLiteral()) {
          // The size of current dimension of current input  is a constant
          outputDims[dim] = currInputBounds.getDim(dim);
        }
      }
    }
  }
  outputDims[axisIndex] = cumulativeAxisSize;

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir
