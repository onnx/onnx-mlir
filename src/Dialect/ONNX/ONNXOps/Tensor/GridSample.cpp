/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ GridSample.cpp - ONNX Operations ------------------===//
//
// Copyright (c) 2024 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file provides definition of ONNX dialect GridSample operation.
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
LogicalResult ONNXGridSampleOpShapeHelper::computeShape() {

  // Read data and indices shapes as dim indices.
  ONNXGridSampleOpAdaptor operandAdaptor(operands);
  DimsExpr inputDims;
  DimsExpr gridDims;
  createIE->getShapeAsDims(operandAdaptor.getX(), inputDims);
  createIE->getShapeAsDims(operandAdaptor.getGrid(), gridDims);

  int64_t gridRank = gridDims.size();

  // Input's dimensions of rank r+2 should be in the form of (N,C,D1,D2,...,Dr)
  // Grid's dimensions should also have rank r+2 and be in the form of
  // (N,D1_out,D2_out,...,Dr_out,r).
  // The output Y will have shape (N, C, D1_out, D2_out, ..., Dr_out).
  DimsExpr outputDims;
  outputDims.emplace_back(inputDims[0]);
  outputDims.emplace_back(inputDims[1]);
  for (int i = 1; i < gridRank - 1; ++i) {
    outputDims.emplace_back(gridDims[i]);
  }

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXGridSampleOp::verify() {
  ONNXGridSampleOpAdaptor operandAdaptor(*this);
  auto op = mlir::cast<ONNXGridSampleOp>(*this);

  const auto alignCorners = op.getAlignCorners();
  if (alignCorners != 0 && alignCorners != 1) {
    return emitOpError("align_corners needs to be 0 or 1");
  }
  const auto mode = op.getMode();
  if (mode != "linear" && mode != "nearest" && mode != "cubic") {
    return emitOpError("mode needs to be linear, nearest or cubic");
  }
  const auto paddingMode = op.getPaddingMode();
  if (paddingMode != "zeros" && paddingMode != "border" &&
      paddingMode != "reflection") {
    return emitOpError("padding_mode needs to be zeros, border or reflection");
  }

  if (!hasShapeAndRank(getOperation()))
    return success();

  auto inputShape =
      mlir::cast<ShapedType>(operandAdaptor.getX().getType()).getShape();
  int64_t inputRank = inputShape.size();
  auto gridShape =
      mlir::cast<ShapedType>(operandAdaptor.getGrid().getType()).getShape();

  // Check whether the ranks of input and grid are valid and are equal.
  // Input's dimensions of rank r+2 should be in the form of (N,C,D1,D2,...,Dr)
  // Grid's dimensions should also have rank r+2 and be in the form of
  // (N,D1_out,D2_out,...,Dr_out,r).
  if (inputShape.size() != gridShape.size()) {
    return emitOpError() << "Input(=" << inputShape.size()
                         << ") and grid(=" << gridShape.size()
                         << ") have different dim sizes.";
  }

  if (inputShape[0] != gridShape[0]) {
    return emitOpError() << "Input and grid must have the same batch value.";
  }

  if (!ShapedType::isDynamic(gridShape.back()) &&
      gridShape.back() != inputRank - 2) {
    return emitOpError() << "Grid last dim must have been '" << inputRank - 2
                         << "' instead of '" << gridShape.back() << "'.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXGridSampleOp::inferShapes(
    std::function<void(Region &)> /*doShapeInference*/) {

  Type elementType = mlir::cast<ShapedType>(getX().getType()).getElementType();
  ONNXGridSampleOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXGridSampleOp>;
} // namespace onnx_mlir
