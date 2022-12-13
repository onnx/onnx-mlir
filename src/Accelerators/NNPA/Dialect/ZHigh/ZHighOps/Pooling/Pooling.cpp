/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Pooling.cpp - ZHigh Operations --------------------===//
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
// MaxPool2DOp

LogicalResult ZHighMaxPool2DOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()))
    return success();

  Builder builder(getContext());
  ZHighMaxPool2DOpAdaptor operandAdaptor(*this);
  ZHighPoolingOpShapeHelper<ZHighMaxPool2DOp, ZHighMaxPool2DOpAdaptor>
      shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh MaxPool2D parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  updateType(getResult(), outputDims, inputType.getElementType(),
      inputType.getEncoding());
  return success();
}

//===----------------------------------------------------------------------===//
// AvgPool2DOp

LogicalResult ZHighAvgPool2DOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()))
    return success();

  Builder builder(getContext());
  ZHighAvgPool2DOpAdaptor operandAdaptor(*this);
  ZHighPoolingOpShapeHelper<ZHighAvgPool2DOp, ZHighAvgPool2DOpAdaptor>
      shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh AvgPool2D parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  updateType(getResult(), outputDims, inputType.getElementType(),
      inputType.getEncoding());
  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
