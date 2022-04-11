/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- CategoryMapper.cpp - Shape Inference for CategoryMapper Op -----===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX CategoryMapper operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXCategoryMapperOpShapeHelper::computeShape(
    ONNXCategoryMapperOpAdaptor operandAdaptor) {
  Value X = operandAdaptor.X();
  MemRefBoundsIndexCapture bounds(X);
  int64_t rank = bounds.getRank();

  DimsExpr outputDims(rank);
  for (int64_t i = 0; i < rank; ++i)
    outputDims[i] = bounds.getDim(i);
  dimsForOutput() = outputDims;

  return success();
}

} // namespace onnx_mlir
