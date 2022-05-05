/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- GatherND.cpp - Shape Inference for GatherND Op ------------===//
//
// This file implements shape inference for the ONNX GatherND Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include <algorithm>

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXGatherNDOpShapeHelper::computeShape(
    ONNXGatherNDOpAdaptor operandAdaptor) {
  Value data = operandAdaptor.data();
  Value indices = operandAdaptor.indices();
  MemRefBoundsIndexCapture dataBounds(data);
  MemRefBoundsIndexCapture indicesBounds(indices);
  DimsExpr dataDims, indicesDims;
  dataBounds.getDimList(dataDims);
  indicesBounds.getDimList(indicesDims);

  int64_t dataRank = dataDims.size();
  int64_t indicesRank = indicesDims.size();
  int64_t b = op->batch_dims();

  assert(indices.getType().isa<ShapedType>() && "Expecting a shaped type");
  auto indicesType = indices.getType().cast<ShapedType>();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  int64_t indicesLastDim = indicesShape[indicesRank - 1];
  int64_t outputRank = dataRank + indicesRank - indicesLastDim - 1 - b;

  // Ensure the operator contraints are statisfied.
  assert(dataRank >= 1 && "dataRank should be >= 1");
  assert(indicesRank >= 1 && "indicesRank should be >= 1");
  assert(b >= 0 && "batch_dim should not be negative");
  assert(b < std::min(dataRank, indicesRank) &&
         "batch_dims must be smaller than the min(dataRank, indicesRank)");
  assert((indicesLastDim >= 1 && indicesLastDim <= dataRank - b) &&
         "indices.shape[-1] must be in the range [1, dataRank - b]");

  // Save the first 'b' dimension of the shape of the 'indices' tensor.
  DimsExpr batchDims;
  for (int64_t i = 0; i < b; ++i)
    batchDims.emplace_back(indicesDims[i]);

  // output.shape = batchDims + list(indices.shape)[b:-1]
  for (int64_t i = 0; i < b; ++i)
    dimsForOutput().emplace_back(batchDims[i]);
  for (int64_t i = b; i < indicesRank - 1; ++i)
    dimsForOutput().emplace_back(indicesDims[i]);

  // When indices.shape[-1] < data_rank - b,
  //   output_shape += list(data.shape)[batch_dims + indices.shape[-1]:]
  if (indicesLastDim < dataRank - b)
    for (int64_t i = b + indicesLastDim; i < dataRank; ++i)
      dimsForOutput().emplace_back(dataDims[i]);

  assert((int64_t)dimsForOutput().size() == outputRank &&
         "Incorrect shape computation");

  return success();
}

} // namespace onnx_mlir
