/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ONNXShapeHelper.inc - help for shapes ---------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// Provide templated implementation of functions, which is included in the
// ONNXShapeHelper.hpp file. The goal is to separate definition and
// implementation. If we were to not inline it, and put it in a cpp file, then
// we would have to add an explicit template instantiation for each usage.
//
//===----------------------------------------------------------------------===//

// Handle shapes for operations with a single output.
template <class SHAPE_HELPER, class OP, class ADAPTOR>
static mlir::LogicalResult shapeHelperInferShapes(
    OP &op, mlir::Type elementType) {
  SHAPE_HELPER shapeHelper(&op);
  ADAPTOR operandAdaptor(op);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return op.emitError("Failed to scan " + OP::getOperationName() +
                        " parameters successfully");

  llvm::SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(), outputDims);

  updateType(op.getResult(), outputDims, elementType);
  return mlir::success();
}

// Handle shapes for operations with multiple outputs.
template <class SHAPE_HELPER, class OP, class ADAPTOR>
static mlir::LogicalResult shapeHelperInferMultipleShapes(
    OP &op, mlir::TypeRange elementTypes) {
  assert(elementTypes.size() == op.getNumResults() &&
         "Incorrect elementTypes size");

  SHAPE_HELPER shapeHelper(&op);
  ADAPTOR operandAdaptor(op);
  if (mlir::failed(shapeHelper.computeShape(operandAdaptor)))
    return op.emitError("Failed to scan " + OP::getOperationName() +
                        " parameters successfully");

  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    llvm::SmallVector<int64_t, 4> outputDims;
    IndexExpr::getShape(shapeHelper.dimsForOutput(i), outputDims);
    updateType(op.getResults()[i], outputDims, elementTypes[i]);
  }
  return mlir::success();
}
