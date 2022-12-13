/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Unstick.cpp - ZHigh Operations --------------------===//
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

void ZHighUnstickOp::build(
    OpBuilder &builder, OperationState &state, Value input) {
  Type resType;
  ShapedType inputType = input.getType().cast<ShapedType>();
  if (hasRankedType(input)) {
    // Compute shape.
    ArrayRef<int64_t> inputShape = inputType.getShape();
    SmallVector<int64_t, 4> resShape(inputShape.begin(), inputShape.end());
    // Direct unstickify from NHWC to NCHW.
    StringAttr layout = convertZTensorDataLayoutToStringAttr(
        builder, getZTensorLayout(input.getType()));
    if (isNHWCLayout(layout)) {
      assert((inputShape.size() == 4) && "Input must have rank 4");
      // NHWC -> NCHW
      resShape[0] = inputShape[0];
      resShape[1] = inputShape[3];
      resShape[2] = inputShape[1];
      resShape[3] = inputShape[2];
    }
    resType = RankedTensorType::get(resShape, inputType.getElementType());
  } else
    resType = UnrankedTensorType::get(inputType.getElementType());
  build(builder, state, resType, input);
}

LogicalResult ZHighUnstickOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(In()))
    return success();

  OpBuilder b(this->getContext());

  StringAttr layout =
      convertZTensorDataLayoutToStringAttr(b, getZTensorLayout(In().getType()));

  ZHighUnstickOpAdaptor operandAdaptor(*this);
  ZHighUnstickOpShapeHelper shapeHelper(this, layout);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh Unstick parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  updateType(getResult(), outputDims, getElementType(In().getType()));
  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
