/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ArgMinMax.cpp - Shape Inference for ArgMax Op -------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX ArgMin & ArgMax operators.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/Diagnostic.hpp"
#include <type_traits>

using namespace mlir;

namespace onnx_mlir {

template <typename OpShapeHelper, typename OpAdaptor>
static LogicalResult computeShape(
    OpShapeHelper &shapeHelper, OpAdaptor &operandAdaptor) {
  static_assert(
      (std::is_same<OpShapeHelper, ONNXArgMinOpShapeHelper>::value &&
          std::is_same<OpAdaptor, ONNXArgMinOpAdaptor>::value) ||
          (std::is_same<OpShapeHelper, ONNXArgMaxOpShapeHelper>::value &&
              std::is_same<OpAdaptor, ONNXArgMaxOpAdaptor>::value),
      "Unexpected template types");

  // Get info about input data operand.
  auto *op = shapeHelper.op;
  Value data = operandAdaptor.data();
  int64_t dataRank = data.getType().cast<ShapedType>().getRank();
  int64_t axisValue = op->axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(data).
  assert(-dataRank <= axisValue && axisValue < dataRank && "axis out of range");

  // Negative axis means values are counted from the opposite side.
  if (axisValue < 0) {
    axisValue = dataRank + axisValue;
    auto builder = mlir::Builder(op->getContext());
    op->axisAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, /*value=*/axisValue, /*isSigned=*/true)));
  }

  // keepdims is a required attribute and should have default value of 1.
  int64_t keepdims = op->keepdims();
  bool isKeepdims = (keepdims == 1);

  // Compute outputDims
  DimsExpr outputDims;
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t reducedRank = isKeepdims ? dataRank : dataRank - 1;
  outputDims.resize(reducedRank);
  for (int64_t i = 0; i < reducedRank; i++) {
    DimIndexExpr dimOutput;
    if (isKeepdims)
      dimOutput = (i != axisValue) ? dataBounds.getDim(i) : LiteralIndexExpr(1);
    else
      dimOutput =
          (i < axisValue) ? dataBounds.getDim(i) : dataBounds.getDim(i + 1);
    outputDims[i] = dimOutput;
  }

  // Save the final result.
  shapeHelper.setOutputDims(outputDims);
  return success();
}

LogicalResult ONNXArgMinOpShapeHelper::computeShape(
    ONNXArgMinOpAdaptor operandAdaptor) {
  return onnx_mlir::computeShape(*this, operandAdaptor);
}

LogicalResult ONNXArgMaxOpShapeHelper::computeShape(
    ONNXArgMaxOpAdaptor operandAdaptor) {
  return onnx_mlir::computeShape(*this, operandAdaptor);
}

} // namespace onnx_mlir
