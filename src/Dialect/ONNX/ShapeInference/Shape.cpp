/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Shape.cpp - Shape Inference for Shape Op --------------===//
//
// This file implements shape inference for the ONNX Shape Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include <tuple>
#include <utility>

using namespace mlir;

namespace onnx_mlir {

namespace {

// If start axis is omitted, the slice starts from axis 0.
// The end axis, if specified, is exclusive (and the returned value will not
// include the size of that axis). If the end axis is omitted, the axes upto the
// last one will be included. Negative axes indicate counting back from the last
// axis. Note that axes will be clipped to the range [0, r-1], where r is the
// rank of the input tensor if they are out-of-range (after adding r in the case
// of negative axis). Thus, specifying any end value > r is equivalent to
// specifying an end value of r, and specifying any start value < -r is
// equivalent to specifying a start value of 0.
int64_t normalize(int64_t value, int64_t rank) {
  if (value < 0)
    value += rank;

  if (value < 0)
    value = 0;

  if (value > rank)
    value = rank;

  return value;
}

// Compute a slice of the input tensor's shape. The slice starts from axis 0.
// The axes up to the last one will be included. Negative axes indicate counting
// back from the last axis.
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

} // namespace

LogicalResult ONNXShapeOpShapeHelper::computeShape(
    ONNXShapeOpAdaptor operandAdaptor) {
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);

  int64_t start;
  int64_t end;
  std::tie(start, end) = getDataShapeBounds(operandAdaptor);

  // Output is the actual number of values (1D)
  dimsForOutput().emplace_back(LiteralIndexExpr(end - start));

  return success();
}

// Compute the data selected by the Shape operator.
DimsExpr computeSelectedData(ONNXShapeOpAdaptor &operandAdaptor) {
  MemRefBoundsIndexCapture dataBounds(operandAdaptor.data());
  int64_t start;
  int64_t end;
  std::tie(start, end) = getDataShapeBounds(operandAdaptor);
  assert(start >= 0 && start <= end && end <= (int64_t)dataBounds.getRank() &&
         "Unexpected bounds");

  DimsExpr selectedData;
  for (int64_t i = start; i < end; ++i)
    selectedData.emplace_back(dataBounds.getDim(i));

  return selectedData;
}

} // namespace onnx_mlir
