/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Clip.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Clip operation.
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

LogicalResult ONNXClipOpShapeHelper::computeShape(
    ONNXClipOpAdaptor operandAdaptor) {
  Value input = operandAdaptor.input();
  MemRefBoundsIndexCapture bounds(input);
  int64_t rank = bounds.getRank();

  DimsExpr outputDims(rank);
  for (int64_t i = 0; i < rank; ++i)
    outputDims[i] = bounds.getDim(i);
  setOutputDims(outputDims);

  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXClipOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Look at input.
  if (!input().getType().isa<RankedTensorType>())
    return success();
  RankedTensorType inputTy = input().getType().cast<RankedTensorType>();
  Type elementType = inputTy.getElementType();
  // Look at optional min.
  if (!min().getType().isa<NoneType>()) {
    // Has a min, make sure its of the right type.
    if (!min().getType().isa<RankedTensorType>())
      return success();
    // And size.
    RankedTensorType minTy = min().getType().cast<RankedTensorType>();
    if (minTy.getElementType() != elementType)
      return emitError("Element type mismatch between input and min tensors");
    if (minTy.getShape().size() != 0)
      return emitError("Min tensor ranked with nonzero size");
  }
  // Look at optional max
  if (!max().getType().isa<NoneType>()) {
    // Has a max, make sure its of the right type.
    if (!max().getType().isa<RankedTensorType>())
      return success();
    // And size.
    RankedTensorType maxTy = max().getType().cast<RankedTensorType>();
    if (maxTy.getElementType() != elementType)
      return emitError("Element type mismatch between input and max tensors");
    if (maxTy.getShape().size() != 0)
      return emitError("Min tensor ranked with nonzero size");
  }

  updateType(getResult(), inputTy.getShape(), elementType);
  return success();
}
