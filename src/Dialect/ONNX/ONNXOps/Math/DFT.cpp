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

template <>
LogicalResult ONNXDFTOpShapeHelper::computeShape() {
  ONNXDFTOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  // Get info about input data operand.
  Value input = operandAdaptor.getInput();
  // Get the rank to compensate for N dimensions.
  int64_t rank = createIE->getShapedTypeRank(input);

  // Axis is a required attribute and should have default value of 1.
  int64_t axis = operandAdaptor.getAxis();

  // OneSided is a required attribute and should have default value of 0.
  // However oneSided can also be a value of 1 and if so a specific shape is
  // expected Values can be 0 or 1.
  int64_t oneSided = operandAdaptor.getOnesided();
  bool isOneSided = (oneSided == 0);

  // Compute outputDims for DFT.
  LiteralIndexExpr one(1);
  DimsExpr outputDims;
  for (int64_t i = 0; i < rank; ++i) {
    if (isOneSided) {
      outputDims.emplace_back(createIE->getShapeAsDim(input, i));
    } else {
      if (axis + 1 == i) {
        IndexExpr d = createIE->getShapeAsDim(input, i).floorDiv(2) + one;
        outputDims.emplace_back(d);
      } else {
        outputDims.emplace_back(createIE->getShapeAsDim(input, i));
      }
    }
  }
  outputDims.emplace_back(LiteralIndexExpr(2));

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
  // Cannot infer the output shape if the input shape is not yet known.
  if (!hasShapeAndRank(getInput()))
    return success();

  Type elementType = getInput().getType().cast<ShapedType>().getElementType();
  ONNXDFTOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXDFTOp>;
} // namespace onnx_mlir
