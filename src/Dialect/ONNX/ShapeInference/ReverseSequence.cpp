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

LogicalResult ONNXReverseSequenceOpShapeHelper::computeShape(
    ONNXReverseSequenceOpAdaptor operandAdaptor) {

  // Get info about input data operand.
  Value input = operandAdaptor.input();
  MemRefBoundsIndexCapture inputBounds(input);
  int64_t inputRank = inputBounds.getRank();

  DimsExpr outputDims;
  for (int64_t i = 0; i < inputRank; ++i)
    outputDims.emplace_back(inputBounds.getDim(i));

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir
