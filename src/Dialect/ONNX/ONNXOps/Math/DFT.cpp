/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- DFT.cpp - Lowering DFT Op --------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect DFT operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <typename OP_TYPE>
LogicalResult ONNXGenericDFTOpShapeHelper<OP_TYPE>::customComputeShape(
    IndexExpr &axis) {
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());

  // Get info about input data operand.
  Value input = operandAdaptor.getInput();
  // Get the rank to compensate for N dimensions.
  if (!hasShapeAndRank(input)) {
    return failure();
  }
  int64_t rank = createIE->getShapedTypeRank(input);

  // Check if the dimension for axis is a literal and in range.
  // TO-FIX: https://github.com/onnx/onnx-mlir/issues/2752
  if (!axis.isLiteral())
    return op->emitError("Can not perform Discrete Fourier Transform on "
                         "dynamic dimensions at this time");

  // OneSided is a required attribute and should have default value of 0.
  // However oneSided can also be a value of 1 and if so a specific shape is
  // expected Values can be 0 or 1. When onesided is 0 it is complex input and
  // when onesided is 1 it is a real input.
  int64_t oneSided = operandAdaptor.getOnesided();
  int64_t axisValue = axis.getLiteral();

  bool isOneSided = (oneSided == 1);
  bool isAxis = (axisValue == 1);

  // Compute outputDims for DFT.

  LiteralIndexExpr one(1);
  DimsExpr outputDims;
  for (int64_t i = 0; i < rank - 1; ++i) {
    if (!isOneSided) { // onesided is 0
      outputDims.emplace_back(createIE->getShapeAsDim(input, i));
    } else { // onesided is 1 and axis is 0
      if (!isAxis && i == 1) {
        IndexExpr d = createIE->getShapeAsDim(input, i).floorDiv(2) + one;
        outputDims.emplace_back(d);
      } else { // onesided is 1 and axis is 1 or onesided is 1 and axis is N
        outputDims.emplace_back(createIE->getShapeAsDim(input, i));
      }
    }
  }
  outputDims.emplace_back(LitIE(2));

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

template <>
LogicalResult ONNXGenericDFTOpShapeHelper<ONNXDFTOp>::computeShape() {
  typename ONNXDFTOp::Adaptor operandAdaptor(operands, op->getAttrDictionary());
  IndexExpr axis = createIE->getIntAsSymbol(operandAdaptor.getAxis());
  return customComputeShape(axis);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// DFT Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXDFTOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer the output shape if the operands shape isn't known yet.
  if (!hasShapeAndRank(getOperation()))
    return success();

  // Not yet shaped, wait for later.
  if (!isNoneValue(getAxis()) && !hasShapeAndRank(getAxis()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getInput().getType()).getElementType();
  ONNXDFTOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXGenericDFTOpShapeHelper<ONNXDFTOp>;
} // namespace onnx_mlir
