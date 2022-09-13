/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Compress.cpp - Shape Inference for Compress Op -----------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX Compress Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/Diagnostic.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXCompressOpShapeHelper::computeShape(
    ONNXCompressOpAdaptor operandAdaptor) {
  // Check that input and condition are ranked.
  Value input = operandAdaptor.input();
  ShapedType inputType = input.getType().dyn_cast_or_null<ShapedType>();
  assert(inputType && inputType.hasRank() &&
         "Input should have a known shape and rank");
  int64_t inputRank = inputType.getRank();
  Value cond = operandAdaptor.condition();
  ShapedType condType = cond.getType().dyn_cast_or_null<ShapedType>();
  assert(condType && condType.hasRank() &&
         "Condition should have a known shape and rank");
  Optional<int64_t> optionalAxis = op->axis();

  // axis attribute (if specified) must be in the range [-r,r-1], where r =
  // rank(input).
  assert((!optionalAxis.has_value() || (-inputRank <= optionalAxis.value() &&
                                           optionalAxis.value() < inputRank)) &&
         "axis out of range");

  // Get the dimension derived from the condition. Assume in shape helper that
  // it is only going to be a question mark. ONNX to Krnl lowering will compute
  // the actual value.
  // TODO: if cond is constant, the compute the actual value.
  IndexExpr dynDim;
  if (scope->isShapeInferencePass())
    dynDim = QuestionmarkIndexExpr(); // Value for runtime dim.
  else
    dynDim = LiteralIndexExpr(-1); // Dummy value to be replaced in lowering.

  // Compute dims for output.
  DimsExpr outputDims;
  if (!optionalAxis.has_value())
    // Reduced to a single dimensional array, of dynamic size.
    outputDims.emplace_back(dynDim);
  else {
    // Has same dimensionality as input, with axis dimension being the dynamic
    // size.
    MemRefBoundsIndexCapture inputBounds(input);
    inputBounds.getDimList(outputDims);

    // Negative axis means values are counted from the opposite side.
    // TODO: should be in normalization pass
    int64_t axisValue = optionalAxis.value();
    if (axisValue < 0)
      axisValue += inputRank;

    outputDims[axisValue] = dynDim;
  }

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir
