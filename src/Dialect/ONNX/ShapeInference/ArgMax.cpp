/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ArgMax.cpp - Shape Inference for ArgMax Op ----------------===//
//
// This file implements shape inference for the ONNX ArgMax Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

using namespace onnx_mlir;

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

  // axis is a required attribute and should have default value of 0.
  int64_t axisValue = op->axis();

  // Accepted axis range is [-r, r-1] where r = rank(data).
  if (axisValue < -1 * (int64_t)dataRank || axisValue >= (int64_t)dataRank) {
    return op->emitError("ArgMax axis value out of bound");
  }

  if (axisValue < 0) {
    axisValue = dataRank + axisValue;
    auto builder = mlir::Builder(op->getContext());
    op->axisAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, /*value=*/axisValue, /*isSigned=*/true)));
  }

  // keepdims is a required attribute and should have default value of 1.
  int64_t keepdims = op->keepdims();
  bool isKeepdims = (keepdims == 1) ? true : false;

  // Compute outputDims
  DimsExpr outputDims;
  MemRefBoundsIndexCapture dataBounds(data);
  int reducedRank = isKeepdims ? dataRank : dataRank - 1;
  outputDims.resize(reducedRank);
  for (auto i = 0; i < reducedRank; i++) {
    DimIndexExpr dimOutput;
    if (isKeepdims) {
      if (i != axisValue)
        dimOutput = dataBounds.getDim(i);
      else
        dimOutput = LiteralIndexExpr(1);
    } else {
      if (i < axisValue)
        dimOutput = dataBounds.getDim(i);
      else
        dimOutput = dataBounds.getDim(i + 1);
    }
    outputDims[i] = dimOutput;
  }

  // Save the final result.
  dimsForOutput(0) = outputDims;
  return success();
}
