/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ StickForLSTM.cpp - ZHigh Operations ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// ShapeHelper
//===----------------------------------------------------------------------===//

LogicalResult ZHighStickForLSTMOpShapeHelper::computeShape() {
  ZHighStickForLSTMOp::Adaptor operandAdaptor(operands);
  Value fGate = operandAdaptor.getFGate();

  // Output dims of result.
  DimsExpr outputDims;

  // Get operands and bounds.
  SmallVector<IndexExpr, 4> fGateDims;
  createIE->getShapeAsDims(fGate, fGateDims);
  int64_t rank = fGateDims.size();

  for (int64_t i = 0; i < rank - 1; ++i)
    outputDims.emplace_back(fGateDims[i]);
  IndexExpr lastDim = fGateDims[rank - 1] * LitIE(4);
  outputDims.emplace_back(lastDim);

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighStickForLSTMOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasRankedType(getFGate()) && !hasRankedType(getIGate()) &&
      !hasRankedType(getCGate()) && !hasRankedType(getOGate()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getResult().getType()).getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::FICO);

  ZHighStickForLSTMOpShapeHelper shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(elementType, encoding);
}

} // namespace zhigh
} // namespace onnx_mlir
