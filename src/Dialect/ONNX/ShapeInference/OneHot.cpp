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

using namespace mlir;

namespace onnx_mlir {

ONNXOneHotOpShapeHelper::ONNXOneHotOpShapeHelper(
    ONNXOneHotOp *newOp, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXOneHotOp>(
          newOp, newOp->getOperation()->getNumResults(), inScope) {}

ONNXOneHotOpShapeHelper::ONNXOneHotOpShapeHelper(ONNXOneHotOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXOneHotOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal, inScope) {}

LogicalResult ONNXOneHotOpShapeHelper::computeShape(
    ONNXOneHotOpAdaptor operandAdaptor) {
  Value indices = operandAdaptor.indices();
  MemRefBoundsIndexCapture indicesBounds(indices);
  int64_t indicesRank = indicesBounds.getRank();

  // Axis is a required attribute and should have default value of -1.
  axis = op->axis();
  if (axis < 0)
    axis += indicesRank + 1;
  assert(axis >= 0 && axis <= indicesRank && "tested in verify");

  Value depthVal = operandAdaptor.depth();
  DenseElementsAttr depthAttr = fGetDenseVal(depthVal);
  if (depthAttr) {
    Type depthType = depthVal.getType();
    int64_t val = getScalarValue<int64_t>(depthAttr, depthType);
    if (val < 1)
      op->emitError("OneHot depth must be greater than 1");
    depth = LiteralIndexExpr(val);
  } else if (scope->isShapeInferencePass()) {
    depth = QuestionmarkIndexExpr();
  } else {
    // Code gen phase, compute the value
    MathBuilder createMath(scope->getRewriter(), op->getLoc());
    Value val = fLoadVal(scope->getRewriter(), op->getLoc(), depthVal, 0);
    // Specs allows depth to be any kind of ints or float. Must transform this
    // to index type as it is used to define data types.
    Value indexVal = createMath.castToIndex(val);
    depth = DimIndexExpr(indexVal);
  }

  // Compute outputDims
  int outputRank = indicesRank + 1;
  DimsExpr outputDims(outputRank);
  for (auto i = 0; i < outputRank; i++) {
    DimIndexExpr dimOutput;
    if (i == axis) {
      dimOutput = depth;
    } else if (i < axis) {
      dimOutput = indicesBounds.getDim(i);
    } else {
      dimOutput = indicesBounds.getDim(i - 1);
    }
    outputDims[i] = dimOutput;
  }

  // Save the final result.
  setOutputDims(outputDims);

  return success();
}

} // namespace onnx_mlir
