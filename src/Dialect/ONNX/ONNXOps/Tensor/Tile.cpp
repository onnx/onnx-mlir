/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Tile.cpp - ONNX Operations ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Tile operation.
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
LogicalResult ONNXTileOpShapeHelper::computeShape() {
  ONNXTileOpAdaptor operandAdaptor(operands);
  // Get info about input data operand.
  Value input = operandAdaptor.input();
  int64_t inputRank = createIE->getShapedTypeRank(input);
  Value repeats = operandAdaptor.repeats();
  // Compute outputDims
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  for (auto i = 0; i < inputRank; i++) {
    IndexExpr dimInput = createIE->getShapeAsDim(input, i);
    IndexExpr repeatsValue = createIE->getIntFromArrayAsSymbol(repeats, i);
    outputDims[i] = dimInput * repeatsValue;
  }
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXTileOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(input()) || !hasShapeAndRank(repeats()))
    return success();

  // 'repeats' tensor is an 1D tensor.
  auto repeatsTensorTy = repeats().getType().cast<RankedTensorType>();
  if (repeatsTensorTy.getShape().size() != 1)
    return emitError("Repeats tensor must have rank one");

  Type elementType = input().getType().cast<ShapedType>().getElementType();
  ONNXTileOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXTileOp>;
} // namespace onnx_mlir
