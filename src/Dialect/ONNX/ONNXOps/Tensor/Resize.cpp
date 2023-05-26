/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Resize.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Resize operation.
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

namespace {
// The yolo4 model uses a float tensor with shape [0] to represent that roi
// or scales is absent (in violation of the spec which says that empty string
// inputs represents absent arguments in the protobuf model representation).
// We work around this by interpreting tensors with shape [0] as an alternative
// way to express that an input is absent.
bool isAbsent(Value input) {
  Type type = input.getType();
  if (ShapedType shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.hasStaticShape({0});
  } else {
    assert(isa<NoneType>(type));
    return true;
  }
}
} // namespace

LogicalResult ONNXResizeOpShapeHelper::computeShape() {
  ONNXResizeOpAdaptor operandAdaptor(operands);
  uint64_t rank = createIE->getShapedTypeRank(operandAdaptor.getX());
  DimsExpr inputDims, outputDims;
  createIE->getShapeAsDims(operandAdaptor.getX(), inputDims);
  bool scalesIsAbsent = isAbsent(operandAdaptor.getScales());

  if (!scalesIsAbsent) {
    // Read and save scales as float.
    createIE->getFloatFromArrayAsNonAffine(operandAdaptor.getScales(), scales);
    if (inputDims.size() != scales.size())
      return op->emitError("expected scales to have the same rank as input");
    // Compute output dims = int(floor(float(input dims) * scales)).
    for (uint64_t i = 0; i < rank; ++i) {
      // Special case for scale == 1.0 as converts are then needed.
      if (scales[i].isLiteralAndIdenticalTo(1.0)) {
        outputDims.emplace_back(inputDims[i]);
      } else {
        IndexExpr floatInputDim = inputDims[i].convertToFloat();
        IndexExpr floatProduct = floatInputDim * scales[i];
        // Formula has a floor, but convert of positive number already rounds
        // toward zero, so skip the floor.
        outputDims.emplace_back(floatProduct.convertToIndex());
      }
    }
  } else {
    // Output size is defined by input `sizes`.
    createIE->getIntFromArrayAsSymbols(operandAdaptor.getSizes(), outputDims);
    if (inputDims.size() != outputDims.size())
      return op->emitError("expected scales to have the same rank as input");
    // Compute scales as float(output dims) / float(input dims).
    for (uint64_t i = 0; i < rank; ++i) {
      IndexExpr floatInputDim = inputDims[i].convertToFloat();
      IndexExpr floatOutputDim = outputDims[i].convertToFloat();
      scales.emplace_back(floatOutputDim / floatInputDim);
    }
  }
  // Save output dims
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXResizeOp::verify() {
  bool scalesIsAbsent = isAbsent(getScales());
  bool sizesIsAbsent = isAbsent(getSizes());
  if (scalesIsAbsent && sizesIsAbsent)
    return emitError("scales() and sizes() cannot both be absent");
  if (!scalesIsAbsent && !sizesIsAbsent)
    return emitError("scales() and sizes() cannot both be defined");

  // TODO: Test the sizes of scales or size to be the same as the rank of X.
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXResizeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()))
    return success();

  Type elementType = getX().getType().cast<RankedTensorType>().getElementType();
  ONNXResizeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
