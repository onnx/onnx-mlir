/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Identity.cpp - ONNX Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Identity operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXIdentityOpShapeHelper::computeShape() {
  ONNXIdentityOpAdaptor operandAdaptor(operands);
  return computeShapeFromOperand(operandAdaptor.input());
}
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXIdentityOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXIdentityOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXIdentityOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  fprintf(stderr, "hi alex 1\n");
  if (!hasShapeAndRank(input()))
    return success();

#if 1
  fprintf(stderr, "hi alex 2\n");
  Type elementType =
      input().getType().cast<RankedTensorType>().getElementType();
  ONNXIdentityOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
#else
  getResult().setType(getOperand().getType());
  return success();
#endif
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXIdentityOp>;
} // namespace onnx_mlir
