/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Pad.cpp - Shape Inference for Pad Op ----------------===//
//
// This file implements shape inference for the ONNX Pad Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

ONNXPadOpShapeHelper::ONNXPadOpShapeHelper(
    ONNXPadOp *newOp, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXPadOp>(
          newOp, newOp->getOperation()->getNumResults(), inScope),
      pads() {}

ONNXPadOpShapeHelper::ONNXPadOpShapeHelper(ONNXPadOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXPadOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal, inScope),
      pads() {}

LogicalResult ONNXPadOpShapeHelper::computeShape(
    ONNXPadOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of results.
  DimsExpr outputDims;

  // Get info about input data operand.
  MemRefBoundsIndexCapture dataBounds(operandAdaptor.data());
  uint64_t dataRank = dataBounds.getRank();

  // Initialize context and results (pads & output)
  pads.resize(2 * dataRank); // pads two sides of each axis.
  outputDims.resize(dataRank);

  // `pads` format is : [x1_begin, x2_begin,...,x1_end, x2_end,...],
  // where
  // - xi_begin: the number of pad values added at the beginning of axis `i`
  // - xi_end: the number of pad values added at the end of axis `i`.
  ArrayValueIndexCapture padsCapture(
      operandAdaptor.pads(), fGetDenseVal, fLoadVal);

  // Calculate output dimension sizes.
  for (uint64_t i = 0; i < dataRank; i++) {
    // Get begin/end pads.
    SymbolIndexExpr padBegin(padsCapture.getSymbol(i));
    SymbolIndexExpr padEnd(padsCapture.getSymbol(i + dataRank));
    if (padBegin.isUndefined() || padEnd.isUndefined())
      return op->emitError("pad parameter could not be processed");
    // Get input dim.
    DimIndexExpr dimInput(dataBounds.getDim(i));

    // Calculation for output size.
    IndexExpr dimOutputFinal = padBegin + dimInput + padEnd;

    // Save results.
    pads[i] = padBegin;
    pads[i + dataRank] = padEnd;
    outputDims[i] = dimOutputFinal;
  }

  // Save the final result.
  dimsForOutput() = outputDims;

  return success();
}

} // namespace onnx_mlir
