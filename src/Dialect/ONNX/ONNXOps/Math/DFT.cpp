/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ DFT.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect DFT operation.
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

LogicalResult ONNXDFTOpShapeHelper::computeShape(
    ONNXDFTOpAdaptor operandAdaptor) {
  // Get info about input data operand.
  Value input = operandAdaptor.input();
  auto inputType = input.getType().cast<ShapedType>();

  // Get the rank to compensate for N dimensions
  int64_t rank = inputType.getRank();

  // axis is a required attribute and should have default value of 1.
  int64_t axis = op->axis();

  
  // onesided is a required attribute and should have default value of 0.
  // However onesided can also be a value of 1 and if so a specific shape is
  // expected Values can be 0 or 1.
  int64_t onesided = op->onesided();
  bool isOneSided = (onesided == 0);

  // Compute outputDims for DFT
  DimsExpr outputDims;
  MemRefBoundsIndexCapture dataBounds(input);
  for (int64_t i = 0; i < rank - 1; i++) {
    if (isOneSided) {
      outputDims.emplace_back(dataBounds.getDim(i));
    } else {
      if (axis + 1 == i) {
        outputDims.emplace_back(dataBounds.getDim(i).floorDiv(2));
      } else {
        outputDims.emplace_back(dataBounds.getDim(i));
      }
    }
    outputDims.emplace_back(LiteralIndexExpr(2));
  }

  // Save the final result.
  setOutputDims(outputDims);

  return success();
}
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXDFTOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape is not yet knwon.
  if (!hasShapeAndRank(input()))
    return success();

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXDFTOpShapeHelper, ONNXDFTOp,
      ONNXDFTOpAdaptor>(*this, elementType);
}
