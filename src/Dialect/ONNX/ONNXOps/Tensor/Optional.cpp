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
  if (type().has_value() != input().getType().isa<NoneType>())
    return emitError(
        "Optional should have either type attribute or input value");
  return success();
}

ONNXOpShapeHelper *ONNXOptionalOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnimplementedOpShapeHelper>(
      op, oper, ieb, scope);
}

LogicalResult ONNXOptionalOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type ty;
  if (auto typeAttr = type()) {
    ty = typeAttr.value();
  } else {
    ty = input().getType();
    // checked in verify()
    assert(!ty.isa<NoneType>() && "type attribute or input value needed");
  }
  getResult().setType(OptType::get(ty));
  return success();
}

//===----------------------------------------------------------------------===//
// OptionalGetElement
//===----------------------------------------------------------------------===//

LogicalResult ONNXOptionalGetElementOp::verify() {
  if (!input().getType().isa<OptType>())
    return emitError("OptionalGetElement input should have optional type");
  return success();
}

ONNXOpShapeHelper *ONNXOptionalGetElementOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnimplementedOpShapeHelper>(
      op, oper, ieb, scope);
}

LogicalResult ONNXOptionalGetElementOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type elementType = input().getType().cast<OptType>().getElementType();
  getResult().setType(elementType);
  return success();
}

//===----------------------------------------------------------------------===//
// OptionalHasElement
//===----------------------------------------------------------------------===//

LogicalResult ONNXOptionalHasElementOp::verify() {
  if (!input().getType().isa<OptType>())
    return emitError("OptionalHasElement input should have optional type");
  return success();
}

ONNXOpShapeHelper *ONNXOptionalHasElementOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXUnimplementedOpShapeHelper>(
      op, oper, ieb, scope);
}

LogicalResult ONNXOptionalHasElementOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Builder builder(getContext());
  Type scalarBoolType = RankedTensorType::get({}, builder.getI1Type());
  getResult().setType(scalarBoolType);
  return success();
}
