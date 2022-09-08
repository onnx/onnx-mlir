/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ Gather.cpp - Shape Inference for Gather Op --------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX Gather Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXGatherOpShapeHelper::computeShape(
    ONNXGatherOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Read data and indices shapes as dim indices.
  MemRefBoundsIndexCapture dataBounds(operandAdaptor.data());
  MemRefBoundsIndexCapture indicesBounds(operandAdaptor.indices());
  DimsExpr dataDims, indicesDims;
  dataBounds.getDimList(dataDims);
  indicesBounds.getDimList(indicesDims);

  int64_t dataRank = dataDims.size();
  int64_t axisIndex = op->axis();
  assert(axisIndex >= -dataRank && axisIndex < dataRank && "Invalid axisIndex");

  // Negative value means counting dimensions from the back.
  axisIndex = (axisIndex < 0) ? axisIndex + dataRank : axisIndex;

  // Output has rank of 'indicesRank + (dataRank - 1).
  // Output shape is constructed from 'input' by:
  //    replacing the dimension at 'axis' in 'input' by the shape of
  //    'indices'.
  DimsExpr outputDims;
  for (int i = 0; i < dataRank; ++i) {
    if (i == axisIndex)
      for (IndexExpr d : indicesDims)
        outputDims.emplace_back(d);
    else
      outputDims.emplace_back(dataDims[i]);
  }

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir
