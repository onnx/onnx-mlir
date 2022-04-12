/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Reduction.cpp - Shape Inference for Reduction Op ------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX Reduction operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXReduceSumOpShapeHelper::computeShape(
    ONNXReduceSumOpAdaptor operandAdaptor) {
  Value axes = operandAdaptor.axes();
  MemRefBoundsIndexCapture bounds(axes);
  int64_t rank = bounds.getRank();

  DimsExpr outputDims(rank);
  for (int64_t i = 0; i < rank; ++i)
    outputDims[i] = bounds.getDim(i);
  dimsForOutput() = outputDims;

  return success();
}

} // namespace onnx_mlir
