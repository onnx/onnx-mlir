/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Compress.cpp - ONNX Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Compress operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXCompressOpShapeHelper::computeShape() {
  // Check that input and condition are ranked.
  ONNXCompressOp compressOp = llvm::cast<ONNXCompressOp>(op);
  ONNXCompressOpAdaptor operandAdaptor(operands);
  Value input = operandAdaptor.getInput();
  Value cond = operandAdaptor.getCondition();
  if (!hasShapeAndRank(input)) {
    return failure();
  }
  int64_t inputRank = createIE->getShapedTypeRank(input);
  createIE->assertHasShapeAndRank(cond);
  std::optional<int64_t> optionalAxis = compressOp.getAxis();

  // axis attribute (if specified) must be in the range [-r,r-1], where r =
  // rank(input).
  if (optionalAxis.has_value())
    assert(-inputRank <= optionalAxis.value() &&
           optionalAxis.value() < inputRank && "axis out of range");

  // Get the dimension derived from the condition. Assume in shape helper that
  // it is only going to be a question mark. ONNX to Krnl lowering will compute
  // the actual value.
  // TODO: if cond is constant, the compute the actual value.
  IndexExpr dynDim = QuestionmarkIndexExpr(/*isFloat*/ false);

  // Compute dims for output.
  DimsExpr outputDims;
  if (!optionalAxis.has_value())
    // Reduced to a single dimensional array, of dynamic size.
    outputDims.emplace_back(dynDim);
  else {
    // Has same dimensionality as input, with axis dimension being the dynamic
    // size.
    createIE->getShapeAsDims(input, outputDims);

    // Negative axis means values are counted from the opposite side.
    // TODO: should be in normalization pass
    int64_t axisValue = optionalAxis.value();
    if (axisValue < 0)
      axisValue += inputRank;

    outputDims[axisValue] = dynDim;
  }

  // Cannot refine shape as we may otherwise loose the dynamic dim.
  setOutputDims(outputDims, /*n*/ 0, /*refineShape*/ false);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXCompressOp::verify() {
  // Cannot check constraints if the shape of the inputs is not yet knwon.
  ONNXCompressOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(getOperation()))
    return success();

  int64_t inputRank = mlir::cast<ShapedType>(getInput().getType()).getRank();
  std::optional<int64_t> optionalAxis = getAxis();

  if (optionalAxis.has_value()) {
    // axis attribute must be in the range [-r,r-1], where r = rank(input).
    int64_t axis = optionalAxis.value();
    if (axis < -inputRank || axis >= inputRank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axis,
          onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));
  }

  int64_t condRank = mlir::cast<ShapedType>(getCondition().getType()).getRank();
  if (condRank != 1)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "condition", condRank,
        onnx_mlir::Diagnostic::Range<int64_t>(1, 1));

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXCompressOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape is not yet knwon.
  if (!hasShapeAndRank(getOperation()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getInput().getType()).getElementType();
  ONNXCompressOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXCompressOp>;
} // namespace onnx_mlir
