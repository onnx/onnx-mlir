/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ArgMax.cpp - Shape Inference for ArgMax Op ----------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX ArgMax Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/Diagnostic.hpp"

using namespace mlir;

namespace onnx_mlir {

ONNXArgMaxOpShapeHelper::ONNXArgMaxOpShapeHelper(ONNXArgMaxOp *newOp)
    : ONNXOpShapeHelper<ONNXArgMaxOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXArgMaxOpShapeHelper::ONNXArgMaxOpShapeHelper(ONNXArgMaxOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXArgMaxOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXArgMaxOpShapeHelper::computeShape(
    ONNXArgMaxOpAdaptor operandAdaptor) {
  // Get info about input data operand.
  Value data = operandAdaptor.data();
  int64_t dataRank = data.getType().cast<ShapedType>().getRank();
  int64_t axisValue = op->axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(data).
  if (axisValue < -dataRank || axisValue >= dataRank)
    return onnx_mlir::Diagnostic::attributeOutOfRange(*op->getOperation(),
        "axis", axisValue,
        onnx_mlir::Diagnostic::Range<int64_t>(-dataRank, dataRank - 1));

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
  dimsForOutput() = outputDims;
  return success();
}

} // namespace onnx_mlir
