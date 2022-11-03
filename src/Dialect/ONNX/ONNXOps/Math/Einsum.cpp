/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Einsum.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Einsum operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/Math/EinsumHelper.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXEinsumOp::verify() {
  einsum::ErrorFn errorFn = [this]() -> mlir::InFlightDiagnostic {
    return this->emitOpError() << "equation '" << this->equation() << "': ";
  };

  ONNXEinsumOpAdaptor operandAdaptor(*this);
  ValueRange inputs = operandAdaptor.Inputs();

  if (failed(einsum::verifyEquation(equation(), inputs.size(), errorFn))) {
    return failure();
  }

  Type firstElementType =
      inputs[0].getType().cast<ShapedType>().getElementType();
  for (Value input : inputs) {
    ShapedType type = input.getType().cast<ShapedType>();
    if (type.getElementType() != firstElementType) {
      return emitOpError() << "different input element types";
    }
  }
  if (!llvm::all_of(inputs, hasShapeAndRank))
    return success(); // Can only infer once operand shapes are known.
  return einsum::verifyShapes(operandAdaptor, errorFn);
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXEinsumOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  ONNXEinsumOpAdaptor operandAdaptor(*this);
  if (!llvm::all_of(operandAdaptor.Inputs(), hasShapeAndRank))
    return success(); // Can only infer once operand shapes are known.

  einsum::ErrorFn errorFn = [this]() {
    return this->emitOpError() << "equation '" << this->equation() << "': ";
  };
  FailureOr<einsum::Shape> shape =
      einsum::inferOutputShape(operandAdaptor, errorFn);
  assert(succeeded(shape) && "any failure should be caught in verify()");
  Type elementType =
      getOperand(0).getType().cast<ShapedType>().getElementType();

  updateType(getResult(), *shape, elementType);
  return success();
}
