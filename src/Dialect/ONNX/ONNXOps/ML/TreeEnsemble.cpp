/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- TreeEnsemble.cpp - ONNX Operations -----------------===//
//
// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
// =============================================================================
//
// This file provides definition of ONNX dialect TreeEnsemble operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXTreeEnsembleOpShapeHelper::computeShape() {
  ONNXTreeEnsembleOpAdaptor operandAdaptor(operands);
  Value X = operandAdaptor.getX();

  DimsExpr inputDims;
  createIE->getShapeAsDims(X, inputDims);

  // Output is [N, n_targets]
  DimsExpr outputDims;
  outputDims.emplace_back(inputDims[0]);

  const auto nTargetsAttr = cast<ONNXTreeEnsembleOp>(op).getNTargetsAttr();
  if (nTargetsAttr)
    outputDims.emplace_back(LitIE(nTargetsAttr.getSInt()));
  else
    outputDims.emplace_back(QuestionmarkIndexExpr(/*isFloat=*/false));

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

LogicalResult ONNXTreeEnsembleOp::inferShapes(
    std::function<void(Region &)> /*doShapeInference*/) {
  if (!hasShapeAndRank(getX()))
    return success();

  const Type elementType = getElementTypeOrSelf(getX().getType());
  ONNXTreeEnsembleOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXTreeEnsembleOp>;
} // namespace onnx_mlir
