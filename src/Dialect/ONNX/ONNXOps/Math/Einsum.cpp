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
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXEinsumOpShapeHelper::computeShape() {
  ONNXEinsumOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  ONNXEinsumOp einsumOp = llvm::cast<ONNXEinsumOp>(op);

  // Infer shape, if success, `*shape` holds the results as a
  // einsum::Shape which is defined as a SmallVector<int64_t, 4>.
  auto errorFn = [&]() {
    return einsumOp.emitOpError()
           << "equation '" << einsumOp.getEquation() << "': ";
  };
  FailureOr<einsum::Shape> shape =
      einsum::inferOutputShape(operandAdaptor, errorFn);
  assert(succeeded(shape) && "any failure should be caught in verify()");

  // Translate shape (ints) into list of IndexExpr literals/questionmarks.
  // Limitation: no dynamic shapes are built here.
  DimsExpr outputDims;
  getIndexExprListFromShape(*shape, outputDims);
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXEinsumOp::verify() {
  auto errorFn = [this]() -> InFlightDiagnostic {
    return this->emitOpError() << "equation '" << this->getEquation() << "': ";
  };

  ONNXEinsumOpAdaptor operandAdaptor(*this);
  ValueRange inputs = operandAdaptor.getInputs();

  if (failed(einsum::verifyEquation(getEquation(), inputs.size(), errorFn))) {
    return failure();
  }

  Type firstElementType =
      mlir::cast<ShapedType>(inputs[0].getType()).getElementType();
  for (Value input : inputs) {
    ShapedType type = mlir::cast<ShapedType>(input.getType());
    if (type.getElementType() != firstElementType) {
      return emitOpError() << "different input element types";
    }
  }
  if (!hasShapeAndRank(getOperation()))
    return success(); // Can only infer once operand shapes are known.
  return einsum::verifyShapes(operandAdaptor, errorFn);
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXEinsumOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  ONNXEinsumOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(getOperation()))
    return success(); // Can only infer once operand shapes are known.

  Type elementType =
      mlir::cast<ShapedType>(getOperand(0).getType()).getElementType();
  ONNXEinsumOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXEinsumOp>;
} // namespace onnx_mlir
