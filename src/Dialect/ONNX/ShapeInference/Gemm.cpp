/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Gemm.cpp - Shape Inference for Gemm Op ----------------===//
//
// This file implements shape inference for the ONNX Gemm Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

ONNXGemmOpShapeHelper::ONNXGemmOpShapeHelper(
    ONNXGemmOp *newOp, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXGemmOp>(
          newOp, newOp->getOperation()->getNumResults(), inScope),
      aDims(), bDims(), cDims(), hasBias(false), cRank(-1) {}

ONNXGemmOpShapeHelper::ONNXGemmOpShapeHelper(ONNXGemmOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXGemmOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal, inScope),
      aDims(), bDims(), cDims(), hasBias(false), cRank(-1) {}

LogicalResult ONNXGemmOpShapeHelper::computeShape(
    ONNXGemmOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of result.
  DimsExpr outputDims;

  // Get info.
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
  MemRefBoundsIndexCapture ABounds(A);
  if (op->transA() == 0) {
    aDims = {ABounds.getDim(0), ABounds.getDim(1)};
  } else {
    aDims = {ABounds.getDim(1), ABounds.getDim(0)};
  }
  // Scan dimensions of B with/without transpose.
  MemRefBoundsIndexCapture BBounds(B);
  if (op->transB() == 0) {
    bDims = {BBounds.getDim(0), BBounds.getDim(1)};
  } else {
    bDims = {BBounds.getDim(1), BBounds.getDim(0)};
  }
  // Set output dims of result, creating a copy of it to be safe.
  outputDims = {aDims[0].deepCopy(), bDims[1].deepCopy()};
  // Bias C can be a (unidirectional) broadcast.
  MemRefBoundsIndexCapture CBounds(C);
  if (hasBias) {
    if (cRank == 0) {
      // Broadcast for scalar: both dims are 1.
      cDims = {LiteralIndexExpr(1), LiteralIndexExpr(1)};
    } else if (cRank == 1) {
      // First dim is the one padded.
      cDims = {LiteralIndexExpr(1), CBounds.getDim(0)};
    } else {
      assert(cRank == 2 && "illegal path");
      cDims = {CBounds.getDim(0), CBounds.getDim(1)};
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
