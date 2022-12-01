/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ OneHot.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect OneHot operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

LogicalResult NewONNXOneHotOpShapeHelper::computeShape() {
  ONNXOneHotOp oneHotOp = llvm::cast<ONNXOneHotOp>(op);
  ONNXOneHotOpAdaptor operandAdaptor(operands);
  Value indices = operandAdaptor.indices();

  // hi alex MemRefBoundsIndexCapture indicesBounds(indices);
  int64_t indicesRank = createIE->getTypeRank(indices);

  // Axis is a required attribute and should have default value of -1.
  axis = oneHotOp->axis();
  if (axis < 0)
    axis += indicesRank + 1;
  assert(axis >= 0 && axis <= indicesRank && "tested in verify");

  depth = createIE->getIntValAsSymbol(operandAdaptor.depth(), 0)
  if (depth.isLiteral()) {
    if (depth.getLiteral() < 1)
    return op->emitError("OneHot depth must be greater than 1");
  } else if (! scope->isShapeInferencePass()) {
    // Convert depth to index
    MathBuilder createMath(scope->getRewriter(), op->getLoc());
    Value convertedVal = createMath.castToIndex(depth.getValue());
    depth = DimIndexExpr(convertedVal);
  }
#if 0
  // Original code.
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
#endif

  // Compute outputDims
  int outputRank = indicesRank + 1;
  DimsExpr outputDims(outputRank);
  for (auto i = 0; i < outputRank; i++) {
    DimIndexExpr dimOutput;
    if (i == axis) {
      dimOutput = depth;
    } else if (i < axis) {
      dimOutput = createIE->getShapeAsDim(indices, i);
    } else {
      dimOutput = createIE->getShapeAsDim(indices, i - 1);
    }
    outputDims[i] = dimOutput;
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}


#if 1 // hi alex remove
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
#endif

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXOneHotOp::verify() {
  ONNXOneHotOpAdaptor operandAdaptor = ONNXOneHotOpAdaptor(*this);
  // Check indices.
  Value indices = operandAdaptor.indices();
  if (hasShapeAndRank(indices)) {
    // Get rank.
    int64_t indicesRank = indices.getType().cast<ShapedType>().getRank();
    // Verify axis.
    int64_t axisValue = axis();
    // Unusually, with a rank of 3, acceptable values are 0 (before first) to 3
    // (after last).
    if (axisValue < 0)
      axisValue += indicesRank + 1;
    if (!(axisValue >= 0 && axisValue <= indicesRank))
      return emitOpError("OneHot axis value is out of range");
  }
  // Check that values is a rank 2 with 2 elements
  Value values = operandAdaptor.values();
  if (hasShapeAndRank(values)) {
    ShapedType valuesShape = values.getType().cast<ShapedType>();
    if (valuesShape.getRank() != 1)
      return emitOpError("OneHot values must be 1D tensor");
    int64_t dim = valuesShape.getDimSize(0);
    if (dim >= 0 && dim != 2)
      return emitOpError("OneHot values must be 1D tensor with 2 elements");
  }
  // Depth is a scalar, check when its a tensor of rank 0 or 1.
  Value depth = operandAdaptor.depth();
  if (hasShapeAndRank(depth)) {
    ShapedType depthShape = depth.getType().cast<ShapedType>();
    if (depthShape.getRank() == 1) {
      int64_t dim = depthShape.getDimSize(0);
      if (dim >= 0 && dim != 1)
        return emitOpError("OneHot depth can be 1D tensor with 1 elements");
    } else {
      if (depthShape.getRank() > 1)
        return emitOpError("OneHot depth must be 0 or 1D tensor");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXOneHotOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!indices().getType().isa<RankedTensorType>())
    return success();

  auto elementType = values().getType().cast<ShapedType>().getElementType();
  NewONNXOneHotOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
