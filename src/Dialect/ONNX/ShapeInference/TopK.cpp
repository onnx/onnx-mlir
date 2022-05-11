/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- TopK.cpp - Shape Inference for TopK Op -----------------===//
//
// This file implements shape inference for the ONNX TopK Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXTopKOpShapeHelper::computeShape(
    ONNXTopKOpAdaptor operandAdaptor) {
  DimsExpr outputDims;

  // Get info about X and K operands.
  Value X = operandAdaptor.X();
  Value K = operandAdaptor.K();
  MemRefBoundsIndexCapture XBounds(X);
  int64_t rank = XBounds.getRank();

  // Axis to compute topk.
  int64_t axis = op->axis();
  axis = axis < 0 ? axis + rank : axis;
  assert(axis >= 0 && axis < rank && "axis is out of bound");

  // K is a scalar tensor storing the number of returned values along the given
  // axis.
  ArrayValueIndexCapture kCapture(K, fGetDenseVal, fLoadVal);
  SymbolIndexExpr kIE(kCapture.getSymbol(0));
  if (kIE.isUndefined())
    return op->emitError("K input parameter could not be processed");

  // If K is literal, it must be less than the axis dimension size.
  if (kIE.isLiteral() && XBounds.getDim(axis).isLiteral())
    if (kIE.getLiteral() >= XBounds.getDim(axis).getLiteral())
      return op->emitError("K value is out of bound");

  for (int64_t i = 0; i < rank; ++i) {
    if (i == axis)
      outputDims.emplace_back(kIE);
    else
      outputDims.emplace_back(XBounds.getDim(i));
  }

  dimsForOutput(0) = outputDims;
  dimsForOutput(1) = outputDims;

  return success();
}

} // namespace onnx_mlir
