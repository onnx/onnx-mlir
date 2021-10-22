/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- OneHot.cpp - Shape Inference for OneHot Op -------------===//
//
// This file implements shape inference for the ONNX OneHot Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

ONNXOneHotOpShapeHelper::ONNXOneHotOpShapeHelper(ONNXOneHotOp *newOp)
    : ONNXOpShapeHelper<ONNXOneHotOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXOneHotOpShapeHelper::ONNXOneHotOpShapeHelper(ONNXOneHotOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXOneHotOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXOneHotOpShapeHelper::ComputeShape(
    ONNXOneHotOpAdaptor operandAdaptor) {
  Value indices = operandAdaptor.indices();
  int64_t indicesRank = indices.getType().cast<ShapedType>().getRank();

  // Axis is a required attribute and should have default value of -1.
  int64_t axisValue = op->axis();
  if (axisValue < -1 * indicesRank - 1 || axisValue > indicesRank) {
    return op->emitError("OneHot axis value is out of range");
  }

  // Negative axis is counting dimension from back
  if (axisValue < 0) {
    axisValue = indicesRank + axisValue + 1;
    auto builder = mlir::Builder(op->getContext());
    op->axisAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, /*value=*/axisValue, /*isSigned=*/true)));
  }

  IndexExpr axisDim = QuestionmarkIndexExpr();
  auto constantDepth = getONNXConstantOp(op->depth());
  if (constantDepth) {
    auto depthTensorTy = op->depth().getType().cast<RankedTensorType>();
    int64_t depthValue = (int64_t)getScalarValue(constantDepth, depthTensorTy);
    axisDim = LiteralIndexExpr(depthValue);
  }

  // Compute outputDims
  int outputRank = indicesRank + 1;
  DimsExpr outputDims(outputRank);
  MemRefBoundsIndexCapture indicesBounds(indices);
  for (auto i = 0; i < outputRank; i++) {
    DimIndexExpr dimOutput;
    if (i == axisValue) {
      dimOutput = axisDim;
    } else if (i < axisValue) {
      dimOutput = indicesBounds.getDim(i);
    } else {
      dimOutput = indicesBounds.getDim(i - 1);
    }
    outputDims[i] = dimOutput;
  }

  // Save the final result.
  dimsForOutput(0) = outputDims;

  return success();
}
