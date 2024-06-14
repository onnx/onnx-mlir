/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Optional.cpp - ONNX Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Optional operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Optional
//===----------------------------------------------------------------------===//

LogicalResult ONNXOptionalOp::verify() {
  if (getType().has_value() != isNoneValue(getInput()))
    return emitError(
        "Optional should have either type attribute or input value");
  return success();
}

LogicalResult ONNXOptionalOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type ty;
  if (auto typeAttr = getType()) {
    ty = typeAttr.value();
  } else {
    ty = getInput().getType();
  }
  getResult().setType(OptType::get(ty));
  return success();
}

//===----------------------------------------------------------------------===//
// OptionalGetElement
//===----------------------------------------------------------------------===//

LogicalResult ONNXOptionalGetElementOp::verify() {
  if (!mlir::isa<OptType>(getInput().getType()))
    return emitError("OptionalGetElement input should have optional type");
  return success();
}

LogicalResult ONNXOptionalGetElementOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type elementType = mlir::cast<OptType>(getInput().getType()).getElementType();
  getResult().setType(elementType);
  return success();
}

//===----------------------------------------------------------------------===//
// OptionalHasElement
//===----------------------------------------------------------------------===//

LogicalResult ONNXOptionalHasElementOp::verify() {
  if (!mlir::isa<OptType>(getInput().getType()))
    return emitError("OptionalHasElement input should have optional type");
  return success();
}

LogicalResult ONNXOptionalHasElementOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Builder builder(getContext());
  Type scalarBoolType = RankedTensorType::get({}, builder.getI1Type());
  getResult().setType(cast<TensorType>(scalarBoolType));
  return success();
}
