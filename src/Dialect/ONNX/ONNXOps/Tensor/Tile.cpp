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

LogicalResult ONNXTileOpShapeHelper::computeShape(
    ONNXTileOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Get info about input data operand.
  Value input = operandAdaptor.input();
  // TOFIX: need to check is_a<ShapedType>?
  int64_t inputRank = input.getType().cast<ShapedType>().getShape().size();
  Value repeats = operandAdaptor.repeats();

  // Compute outputDims
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  MemRefBoundsIndexCapture inputBounds(input);
  ArrayValueIndexCapture repeatsCapture(repeats, fGetDenseVal, fLoadVal);
  for (auto i = 0; i < inputRank; i++) {
    DimIndexExpr dimInput(inputBounds.getDim(i));
    SymbolIndexExpr repeatsValue(repeatsCapture.getSymbol(i));
    IndexExpr dimOutput = dimInput * repeatsValue;
    outputDims[i] = dimOutput;
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
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!input().getType().isa<RankedTensorType>())
    return success();

  // Read 'repeats' value.
  if (!repeats().getType().isa<RankedTensorType>())
    return success();

  // 'repeats' tensor is an 1D tensor.
  auto repeatsTensorTy = repeats().getType().cast<RankedTensorType>();
  if (repeatsTensorTy.getShape().size() != 1)
    return emitError("Repeats tensor must have rank one");

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXTileOpShapeHelper, ONNXTileOp,
      ONNXTileOpAdaptor>(*this, elementType);
}
