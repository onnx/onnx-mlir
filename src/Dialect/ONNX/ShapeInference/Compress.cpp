/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- Compress.cpp - Shape Inference for Compress Op ---===//
//
// This file implements shape inference for the ONNX Compress Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

ONNXCompressOpShapeHelper::ONNXCompressOpShapeHelper(ONNXCompressOp *newOp)
    : ONNXOpShapeHelper<ONNXCompressOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXCompressOpShapeHelper::ONNXCompressOpShapeHelper(ONNXCompressOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXCompressOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

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
         "Condition should have a known and rank");
  // Get the dimension derived from the condition. Assume in shape helper
  // that it is only going to be a question mark. ONNX to Krnl lowering will
  // compute the actual value.
  // TODO: if cond is constant, the compute the actual value.
  IndexExpr dynDim;
  if (scope->isShapeInferencePass())
    dynDim = QuestionmarkIndexExpr(); // Value for runtime dim.
  else
    dynDim = LiteralIndexExpr(-1); // Dummy value to be replaced in lowering.
  // Get axis. Value -1 signify axis was not specified. Verifier already checked
  // that the axis, if given, is in range.
  axis = -1;
  if (op->axis().hasValue()) {
    axis = op->axis().getValue();
    if (axis < 0)
      axis += inputRank;
  }
  // Compute dims for output.
  DimsExpr outputDims;
  if (axis == -1) {
    // Reduced to a single dimensional array, of dynamic size.
    outputDims.emplace_back(dynDim);
  } else {
    // Has same dimensionality as input, with axis dimension being the dynamic
    // size.
    MemRefBoundsIndexCapture inputBounds(input);
    inputBounds.getDimList(outputDims);
    outputDims[axis] = dynDim;
  }
  dimsForOutput(0) = outputDims;
  return success();
}

