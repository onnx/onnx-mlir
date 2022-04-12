/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Shape.cpp - Shape Inference for Shape Op --------------===//
//
// This file implements shape inference for the ONNX Shape Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include <utility>

using namespace mlir;

namespace onnx_mlir {

// Compute a slice of the input tensor's shape. The slice starts from axis 0.
// The axes upto the last one will be included. Negative axes indicate counting
// back from the last axis.
static std::pair<int64_t, int64_t> getDataShapeBounds(
    ONNXShapeOpAdaptor &operandAdaptor) {
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t dataRank = dataBounds.getRank();

  // Compute the normalized start/end. Negative value means counting
  // dimensions from the back.
  int64_t normalizedStart = 0;
  int64_t normalizedEnd = dataRank;

  if (normalizedStart < 0)
    normalizedStart += dataRank;
  if (normalizedEnd < 0)
    normalizedEnd += dataRank;

  return std::make_pair(normalizedStart, normalizedEnd);
}

LogicalResult ONNXShapeOpShapeHelper::computeShape(
    ONNXShapeOpAdaptor operandAdaptor) {
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t dataRank = dataBounds.getRank();
  std::pair<int64_t, int64_t> bounds = getDataShapeBounds(operandAdaptor);

  if (bounds.first < 0 || bounds.first > dataRank)
    return op->emitError("start value is out of bound");
  if (bounds.second < 0 || bounds.second > dataRank)
    return op->emitError("end value is out of bound");

  // Output is the actual number of values (1D)
  dimsForOutput().emplace_back(LiteralIndexExpr(bounds.second - bounds.first));
  return success();
}

// Compute the data selected by the Shape operator.
DimsExpr computeSelectedData(ONNXShapeOpAdaptor &operandAdaptor) {
  MemRefBoundsIndexCapture dataBounds(operandAdaptor.data());
  std::pair<int64_t, int64_t> bounds = getDataShapeBounds(operandAdaptor);
  assert(bounds.first >= 0 && bounds.first <= bounds.second &&
         bounds.second <= (int64_t)dataBounds.getRank() && "Unexpected bounds");

  DimsExpr selectedData;
  for (int64_t i = bounds.first; i < bounds.second; ++i)
    selectedData.emplace_back(dataBounds.getDim(i));

  return selectedData;
}

} // namespace onnx_mlir
