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

LogicalResult ZHighStickForLSTMOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(f_gate()) && !hasRankedType(i_gate()) &&
      !hasRankedType(c_gate()) && !hasRankedType(o_gate()))
    return success();

  ZHighStickForLSTMOpAdaptor operandAdaptor(*this);
  ZHighStickForLSTMOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError(
        "Failed to scan ZHigh StickForLSTM parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  Type elementType = getResult().getType().cast<ShapedType>().getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::FICO);
  updateType(getResult(), outputDims, elementType, encoding);
  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
