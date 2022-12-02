/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Gemm.cpp - ONNX Operations ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Gemm operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

NewONNXGemmOpShapeHelper::NewONNXGemmOpShapeHelper(Operation *op,
    ArrayRef<Value> operands, IndexExprBuilder *ieBuilder,
    IndexExprScope *scope)
    : NewONNXOpShapeHelper(op, operands, ieBuilder, scope), aDims(), bDims(),
      cDims(), hasBias(/*dummy value*/ false), cRank(-1) {}

LogicalResult NewONNXGemmOpShapeHelper::computeShape() {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of result.
  DimsExpr outputDims;

  // Get info.
  ONNXGemmOp gemmOp = llvm::cast<ONNXGemmOp>(op);
  ONNXGemmOpAdaptor operandAdaptor(operands);
  Value A = operandAdaptor.A();
  Value B = operandAdaptor.B();
  Value C = operandAdaptor.C();
  hasBias = !C.getType().isa<NoneType>();

  // Test ranks.
  if (A.getType().cast<ShapedType>().getShape().size() != 2)
    return op->emitError("Gemm with A should be a 2D tensor");
  if (B.getType().cast<ShapedType>().getShape().size() != 2)
    return op->emitError("Gemm with B should be a 2D tensor");
  cRank = 0;
  if (hasBias) {
    cRank = C.getType().cast<ShapedType>().getShape().size();
    if (cRank > 2)
      return op->emitError("Gemm with C should be a 1D or 2D tensor");
  }
  // Scan dimensions of A with/without transpose.
  if (gemmOp.transA() == 0) {
    aDims = {createIE->getShapeAsDim(A, 0), createIE->getShapeAsDim(A, 1)};
  } else {
    aDims = {createIE->getShapeAsDim(A, 1), createIE->getShapeAsDim(A, 0)};
  }
  // Scan dimensions of B with/without transpose.
  if (gemmOp.transB() == 0) {
    bDims = {createIE->getShapeAsDim(B, 0), createIE->getShapeAsDim(B, 1)};
  } else {
    bDims = {createIE->getShapeAsDim(B, 1), createIE->getShapeAsDim(B, 0)};
  }
  // Set output dims of result, creating a copy of it to be safe.
  outputDims = {aDims[0].deepCopy(), bDims[1].deepCopy()};
  // Bias C can be a (unidirectional) broadcast.
  if (hasBias) {
    if (cRank == 0) {
      // Broadcast for scalar: both dims are 1.
      cDims = {LiteralIndexExpr(1), LiteralIndexExpr(1)};
    } else if (cRank == 1) {
      // First dim is the one padded.
      cDims = {LiteralIndexExpr(1), createIE->getShapeAsDim(C, 0)};
    } else {
      assert(cRank == 2 && "illegal path");
      cDims = {createIE->getShapeAsDim(C, 0), createIE->getShapeAsDim(C, 1)};
    }
  }
  // Check static dimensions, if we can.
  if (aDims[1].isLiteral() && bDims[0].isLiteral() &&
      aDims[1].getLiteral() != bDims[0].getLiteral()) {
    return op->emitError("Gemm 2nd dim of A is different than 1st dim of B");
  }
  if (hasBias) {
    // Check first dim.
    if (outputDims[0].isLiteral() && cDims[0].isLiteral()) {
      if (cDims[0].getLiteral() == 1 ||
          cDims[0].getLiteral() == outputDims[0].getLiteral()) {
        // We are fine.
      } else {
        return op->emitError("bias add has bad dimension on first dim");
      }
    }
    // Check second dim.
    if (outputDims[1].isLiteral() && cDims[1].isLiteral()) {
      if (cDims[1].getLiteral() == 1 ||
          cDims[1].getLiteral() == outputDims[1].getLiteral()) {
        // We are fine.
      } else {
        return op->emitError("bias add has bad dimension on second dim");
      }
    }
  }
  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXGemmOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  bool hasBias = !C().getType().isa<NoneType>();
  // Cannot infer shape if no shape exists.
  if (!A().getType().isa<RankedTensorType>() ||
      !B().getType().isa<RankedTensorType>() ||
      (hasBias && !C().getType().isa<RankedTensorType>()))
    return success();

  auto elementType = A().getType().cast<ShapedType>().getElementType();

  NewONNXGemmOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
