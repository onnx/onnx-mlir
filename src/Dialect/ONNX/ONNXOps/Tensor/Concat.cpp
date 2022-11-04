/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Concat.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Concat operation.
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

LogicalResult ONNXConcatOpShapeHelper::computeShape(
    ONNXConcatOpAdaptor operandAdaptor) {
  unsigned numInputs = op->getNumOperands();
  Value firstInput = operandAdaptor.inputs().front();
  ArrayRef<int64_t> commonShape =
      firstInput.getType().cast<ShapedType>().getShape();
  int64_t commonRank = commonShape.size();
  int64_t axisIndex = op->axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(inputs).
  assert(-commonRank <= axisIndex && axisIndex < commonRank &&
         "axis out of range");

  // Negative axis means values are counted from the opposite side.
  // TOFIX should be in normalization pass
  if (axisIndex < 0)
    axisIndex += commonRank;

  // For Concat Op, the size of each dimension of inputs should be the same,
  // except for concatenated dimension. To simplify the result, constant
  // size is used if there is one. Otherwise, the dimension of the first
  // input tensor (implementation dependent) is used for the output tensor.
  DimsExpr outputDims(commonRank);
  MemRefBoundsIndexCapture firstInputBounds(operandAdaptor.inputs()[0]);
  for (unsigned dim = 0; dim < commonRank; dim++) {
    outputDims[dim] = firstInputBounds.getDim(dim);
  }
  IndexExpr cumulativeAxisSize =
      DimIndexExpr(firstInputBounds.getDim(axisIndex));

  // Handle the rest of input
  for (unsigned i = 1; i < numInputs; ++i) {
    Value currentInput = operandAdaptor.inputs()[i];
    MemRefBoundsIndexCapture currInputBounds(currentInput);
    for (unsigned dim = 0; dim < commonRank; dim++) {
      if (dim == axisIndex) {
        DimIndexExpr currentSize(currInputBounds.getDim(axisIndex));
        cumulativeAxisSize = cumulativeAxisSize + currentSize;
      } else {
        if (currInputBounds.getDim(dim).isLiteral()) {
          // The size of current dimension of current input  is a constant
          outputDims[dim] = currInputBounds.getDim(dim);
        }
      }
    }
  }
  outputDims[axisIndex] = cumulativeAxisSize;

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXConcatOp::verify() {
  // Cannot verify semantics if the operands do not have a known shape yet.
  ONNXConcatOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  auto commonType =
      operandAdaptor.getOperands().front().getType().cast<ShapedType>();
  ArrayRef<int64_t> commonShape = commonType.getShape();
  int64_t commonRank = commonShape.size();
  int64_t axisIndex = axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(inputs).
  if (axisIndex < -commonRank || axisIndex >= commonRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-commonRank, commonRank - 1));

  if (axisIndex < 0)
    axisIndex += commonRank;

  // All input tensors must have the same shape, except for the dimension size
  // of the axis to concatenate on.
  for (Value operand : operandAdaptor.getOperands()) {
    ArrayRef<int64_t> operandShape =
        operand.getType().cast<ShapedType>().getShape();
    int64_t operandRank = operandShape.size();
    if (operandRank != commonRank)
      return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
          *this->getOperation(), operand, operandRank,
          std::to_string(commonRank));

    for (int64_t dim = 0; dim < commonRank; ++dim) {
      if (dim == axisIndex)
        continue;
      if (operandShape[dim] != -1 && commonShape[dim] != -1 &&
          operandShape[dim] != commonShape[dim])
        return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
            *this->getOperation(), operand, dim, operandShape[dim],
            std::to_string(commonShape[dim]));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXConcatOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // The check of constraints is kept
  // However, current check handles dynamic dim only for the concat dim
  int inputNum = getNumOperands();
  for (int i = 0; i < inputNum; ++i) {
    if (!getOperand(i).getType().isa<RankedTensorType>())
      return success();
  }
  // Checking value of axis parameter.
  auto commonType = getOperand(0).getType().cast<RankedTensorType>();
  auto commonShape = commonType.getShape();
  int64_t commonRank = commonShape.size();
  int64_t axisIndex = axis();
  // Negative axis means values are counted from the opposite side.
  if (axisIndex < 0) {
    axisIndex = commonRank + axisIndex;
    // Tong Chen:
    // TOFIX: attribute modification should be into canonicalization
    // I did not move the code into ShapeHelper
    auto builder = mlir::Builder(getContext());
    axisAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, /*value=*/axisIndex, /*isSigned=*/true)));
  }

  return shapeHelperInferShapes<ONNXConcatOpShapeHelper, ONNXConcatOp,
      ONNXConcatOpAdaptor>(*this, commonType.getElementType());
}
