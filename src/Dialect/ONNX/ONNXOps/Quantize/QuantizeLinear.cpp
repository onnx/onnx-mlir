/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ QuantizeLinear.cpp.cpp - ONNX Operations ----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect QuantizeLinear operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXQuantizeLinearOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto inTy = x().getType().dyn_cast<RankedTensorType>();
  if (!inTy) {
    return success();
  }

  auto yTy = y().getType().cast<ShapedType>();

  if (!yTy.hasStaticShape()) {
    Value zero = y_zero_point();

    Type elementType;
    if (isFromNone(zero)) {
      // If zero point type isn't provided, output type defaults to ui8.
      elementType = IntegerType::get(getContext(), 8, IntegerType::Unsigned);
    } else {
      // If zero point is provided, output type is same as zero point type.
      elementType = zero.getType().cast<ShapedType>().getElementType();
    }
    RankedTensorType outType =
        RankedTensorType::get(inTy.getShape(), elementType);
    y().setType(outType);
  }

  return success();
}
