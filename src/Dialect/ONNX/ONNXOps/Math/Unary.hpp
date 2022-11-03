/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Unary.hpp - ONNX Operations -----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides support for unary operations (Elementwise or not).
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

/// Handle shape inference for unary element-wise operators.
LogicalResult inferShapeForUnaryOps(Operation *op) {
  Value input = op->getOperand(0);
  Value output = op->getResult(0);

  if (!hasShapeAndRank(input))
    return success();

  ONNXGenericOpUnaryShapeHelper shapeHelper(op);
  if (failed(shapeHelper.computeShape(input)))
    return op->emitError("Failed to scan parameters successfully");
  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(), outputDims);

  // Inferred shape is getting from the input's shape.
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  updateType(
      output, outputDims, inputType.getElementType(), inputType.getEncoding());
  return success();
}

} // namespace onnx_mlir
