/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Dropout.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Dropout operation.
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
LogicalResult ONNXDropoutOpShapeHelper::computeShape() {
  ONNXDropoutOp dropout = llvm::cast<ONNXDropoutOp>(op);
  ONNXDropoutOpAdaptor operandAdaptor(operands, op->getAttrDictionary());

  // First dim is the same as data.
  DimsExpr outputDims;
  createIE->getShapeAsDims(operandAdaptor.getData(), outputDims);
  setOutputDims(outputDims, 0);
  // Optional Mask has also the same size as data. If none, size is empty.
  if (isNoneValue(dropout.getMask()))
    outputDims.clear();
  setOutputDims(outputDims, 1);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXDropoutOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getData()))
    return success();

  Type outputElementType =
      mlir::cast<RankedTensorType>(getData().getType()).getElementType();
  IntegerType maskElementType =
      IntegerType::get(getContext(), 1, IntegerType::Signless);
  ONNXDropoutOpShapeHelper shapeHelper(getOperation(), {});
  // Mask is optional, meaning its type may be None. If that is the case,
  // computeShapeAndUpdateTypes will not override that none type with an array
  // of boolean.
  return shapeHelper.computeShapeAndUpdateTypes(
      {outputElementType, maskElementType});
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXDropoutOp>;
} // namespace onnx_mlir
