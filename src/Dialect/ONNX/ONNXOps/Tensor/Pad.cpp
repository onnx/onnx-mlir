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
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include <numeric>

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
  Value axesOperand = operandAdaptor.getAxes();

  uint64_t dataRank = createIE->getShapedTypeRank(dataOperand);

  bool isFloat = isa<FloatType>(getElementType(dataOperand.getType()));
  // Initially, output dim sizes are all unknown.
  DimsExpr outputDims(dataRank, QuestionmarkIndexExpr(/*IsFloat=*/isFloat));

  // Compute the values of the "axes" array. If "axes" operand is not provided,
  // it is a range from 0 to dataRank. If it is provided, it is a list of
  // integers and the values must be in the range [-dataRank, dataRank).
  SmallVector<uint64_t> axes;
  if (isNoneValue(axesOperand)) {
    axes.resize(dataRank);
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    auto axesSize = createIE->getArraySize(axesOperand);

    // Bail out: If axes is dynamic, output is also dynamic.
    if (axesSize == ShapedType::kDynamic) {
      setOutputDims(outputDims);
      return success();
    }

    if (axesSize < 0) {
      return op->emitError("axes size must be greater than 0");
    }

    // Iterate over axesOperand to figure out the axes that will be padded
    for (auto axesOperandIndex : llvm::seq(axesSize)) {
      IndexExpr padsAxis =
          createIE->getIntFromArrayAsSymbol(axesOperand, axesOperandIndex);

      // If the values of axesOperand cannot be calculated at compile time, bail
      // out...
      if (!padsAxis.isLiteral()) {
        setOutputDims(outputDims);
        return success();
      }

      int64_t positiveAxis = padsAxis.getLiteral();
      if (positiveAxis < 0) {
        positiveAxis += dataRank;
      }

      if (positiveAxis + (int)dataRank < 0 || positiveAxis >= (int)dataRank) {
        return op->emitError("axes value is out of bounds");
      }

      axes.push_back(positiveAxis);
    }
  }

  // Initialize pads according to the most likely case
  pads.resize(2 * dataRank); // pads two sides of each axis.

  llvm::SmallSet<uint64_t, 4> visited;
  for (auto [idx, axis] : llvm::enumerate(axes)) {
    // `pads` format is : [x1_begin, x2_begin,...,x1_end, x2_end,...],
    // where
    // - xi_begin: the number of pad values added at the beginning of axis `i`
    // - xi_end: the number of pad values added at the end of axis `i`.
    // Get begin/end pads.
    SymbolIndexExpr padBegin(
        createIE->getIntFromArrayAsSymbol(padsOperand, idx));
    SymbolIndexExpr padEnd(
        createIE->getIntFromArrayAsSymbol(padsOperand, idx + axes.size()));

    if (padBegin.isUndefined() || padEnd.isUndefined()) {
      return op->emitError("pad parameter could not be processed");
    }

    // Get input dim.
    DimIndexExpr dimInput(createIE->getShapeAsDim(dataOperand, axis));

    // Calculation for output size.
    IndexExpr dimOutputFinal = (padBegin + dimInput) + padEnd;

    visited.insert(axis);

    // Currently "pads" is only used when axes is NoneType and for constant
    // propagation
    if (isNoneValue(axesOperand)) {
      pads[axis] = padBegin;
      pads[axis + dataRank] = padEnd;
    }

    outputDims[axis] = dimOutputFinal;
  }

  for (auto i : llvm::seq(dataRank)) {
    if (!visited.count(i)) {
      outputDims[i] = createIE->getShapeAsLiteral(dataOperand, i);
    }
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
  ShapedType dataTy = getData().getType().cast<ShapedType>();
  Type constTy = getConstantValue().getType();

  if (!isNoneValue(getConstantValue())) {
    // Check that the constant has the same element type as the input
    ShapedType shapedConstTy = constTy.cast<ShapedType>();
    if (dataTy.getElementType() != shapedConstTy.getElementType()) {
      return emitOpError("Pad with constant_value that doesn't match the "
                         "element type of the input.");
    }
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

  Type elementType = getData().getType().cast<ShapedType>().getElementType();

  ONNXPadOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
