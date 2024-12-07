/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Pad.cpp - ONNX Operations -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Pad operation.
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

LogicalResult ONNXPadOpShapeHelper::computeShape() {
  ONNXPadOpAdaptor operandAdaptor(operands);
  Value dataOperand = operandAdaptor.getData();
  Value padsOperand = operandAdaptor.getPads();
  DimsExpr outputDims;

  // Get info about input data operand.
  if (!hasShapeAndRank(dataOperand)) {
    return failure();
  }
  uint64_t dataRank = createIE->getShapedTypeRank(dataOperand);

  // Initialize context and results (pads & output)
  pads.resize(2 * dataRank); // pads two sides of each axis.
  outputDims.resize(dataRank);

  // `pads` format is : [x1_begin, x2_begin,...,x1_end, x2_end,...],
  // where
  // - xi_begin: the number of pad values added at the beginning of axis `i`
  // - xi_end: the number of pad values added at the end of axis `i`.

  // Calculate output dimension sizes.
  for (uint64_t i = 0; i < dataRank; i++) {
    // Get begin/end pads.
    SymbolIndexExpr padBegin(createIE->getIntFromArrayAsSymbol(padsOperand, i));
    SymbolIndexExpr padEnd(
        createIE->getIntFromArrayAsSymbol(padsOperand, i + dataRank));
    if (padBegin.isUndefined() || padEnd.isUndefined())
      return op->emitError("pad parameter could not be processed");
    // Get input dim.
    DimIndexExpr dimInput(createIE->getShapeAsDim(dataOperand, i));

    // Calculation for output size.
    IndexExpr dimOutputFinal = (padBegin + dimInput) + padEnd;

    // Save results.
    pads[i] = padBegin;
    pads[i + dataRank] = padEnd;
    outputDims[i] = dimOutputFinal;
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadOp::verify() {
  ShapedType dataTy = mlir::cast<ShapedType>(getData().getType());
  Type constTy = getConstantValue().getType();

  if (!isNoneValue(getConstantValue())) {
    // Check that the constant has the same element type as the input
    ShapedType shapedConstTy = mlir::cast<ShapedType>(constTy);
    if (dataTy.getElementType() != shapedConstTy.getElementType()) {
      return emitOpError("Pad with constant_value that doesn't match the "
                         "element type of the input.");
    }
  }

  if (!isNoneValue(getAxes())) {
    return emitOpError("Axes input is not currently supported");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getData()) || !hasShapeAndRank(getPads()))
    return success();

  Type elementType =
      mlir::cast<ShapedType>(getData().getType()).getElementType();

  ONNXPadOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
