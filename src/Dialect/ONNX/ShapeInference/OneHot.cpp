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
  MemRefBoundsIndexCapture indicesBounds(indices);
  int64_t indicesRank = indicesBounds.getRank();

  // Axis is a required attribute and should have default value of -1.
  axis = op->axis();
  if (axis < 0)
      axis += indicesRank + 1;
  assert(axis >=0 && axis<=indicesRank && "tested in verify");

  IndexExpr axisDim = QuestionmarkIndexExpr();
  auto constantDepth = getONNXConstantOp(op->depth());
  if (constantDepth) {
    auto depthTensorTy = op->depth().getType().cast<RankedTensorType>();
    int64_t depthValue = getScalarValue<int64_t>(constantDepth, depthTensorTy);
    axisDim = LiteralIndexExpr(depthValue);
  }

  // Compute outputDims
  int outputRank = indicesRank + 1;
  DimsExpr outputDims(outputRank);
  for (auto i = 0; i < outputRank; i++) {
    DimIndexExpr dimOutput;
    if (i == axis) {
      dimOutput = axisDim;
    } else if (i < axis) {
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
