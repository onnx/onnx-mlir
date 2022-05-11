/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- GatherElements.cpp - Shape Inference for GatherElements Op -----===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX GatherElements Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXGatherElementsOpShapeHelper::computeShape(
    ONNXGatherElementsOpAdaptor operandAdaptor) {
  MemRefBoundsIndexCapture indicesBounds(operandAdaptor.indices());
  DimsExpr indicesDims;
  indicesBounds.getDimList(indicesDims);

  // Output has the shape of indices.
  int64_t indicesRank = indicesDims.size();
  for (int i = 0; i < indicesRank; ++i)
    dimsForOutput().emplace_back(indicesDims[i]);

  return success();
}

} // namespace onnx_mlir
