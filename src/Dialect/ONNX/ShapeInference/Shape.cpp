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
#include <tuple>

using namespace mlir;

namespace onnx_mlir {

// Compute a slice of the input tensor's shape. The slice starts from axis 0.
// The axes up to the last one will be included. Negative axes indicate counting
// back from the last axis.
namespace {

int64_t normalize(int64_t value, int64_t rank) {

  if (value < 0)
    value += rank;

  if (value < 0)
    value = 0;

  if (value > rank)
    value = rank;

  return value;

}

std::pair<int64_t, int64_t> getDataShapeBounds(
    ONNXShapeOpAdaptor &operandAdaptor) {
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t rank = dataBounds.getRank();

  // Compute the normalized start/end. Negative value means counting
  // dimensions from the back.
  int64_t start = operandAdaptor.start();
  int64_t end = rank;
  if (operandAdaptor.end().has_value()) {
    end = operandAdaptor.end().value();
  }

  return std::make_pair(normalize(start, rank), normalize(end, rank));
}

}

LogicalResult ONNXShapeOpShapeHelper::computeShape(
    ONNXShapeOpAdaptor operandAdaptor) {
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);

  int64_t first;
  int64_t second;
  std::tie(first, second) = getDataShapeBounds(operandAdaptor);

  // Output is the actual number of values (1D)
  dimsForOutput().emplace_back(LiteralIndexExpr(second - first));

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
