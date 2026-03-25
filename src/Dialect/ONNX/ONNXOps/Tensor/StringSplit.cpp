/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- StringSplit.cpp - ONNX Operations ------------------===//
//
// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
// =============================================================================
//
// This file provides definition of ONNX dialect StringSplit operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXStringSplitOpShapeHelper::computeShape() {
  ONNXStringSplitOpAdaptor operandAdaptor(operands);
  Value X = operandAdaptor.getX();

  DimsExpr inputDims;
  createIE->getShapeAsDims(X, inputDims);

  // Output Y: input_shape + unknown dimension for substrings
  DimsExpr yDims(inputDims);
  yDims.emplace_back(QuestionmarkIndexExpr(/*isFloat=*/false));
  setOutputDims(yDims, 0);

  // Output Z: same shape as input
  setOutputDims(inputDims, 1);
  return success();
}

} // namespace onnx_mlir

LogicalResult ONNXStringSplitOp::inferShapes(
    std::function<void(Region &)> /*doShapeInference*/) {
  if (!hasShapeAndRank(getX()))
    return success();

  const auto stringType = ONNXStringType::get(getContext());
  const auto int64Type = IntegerType::get(getContext(), 64);

  ONNXStringSplitOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateTypes({stringType, int64Type});
}

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXStringSplitOp>;
} // namespace onnx_mlir
