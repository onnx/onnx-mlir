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

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXConcatOpShapeHelper::computeShape() {
  ONNXConcatOp concatOp = mlir::dyn_cast<ONNXConcatOp>(op);
  ONNXConcatOpAdaptor operandAdaptor(operands);

  // Has an empty input. Compute shape later once the empty input is removed by
  // canonicalization.
  for (Value operand : operandAdaptor.getOperands()) {
    ArrayRef<int64_t> operandShape =
        mlir::cast<ShapedType>(operand.getType()).getShape();
    if (operandShape.size() == 1 && operandShape[0] == 0) {
      return success();
    }
  }

  unsigned numInputs = op->getNumOperands();
  Value firstInput = operandAdaptor.getInputs().front();
  ArrayRef<int64_t> commonShape =
      mlir::cast<ShapedType>(firstInput.getType()).getShape();
  int64_t commonRank = commonShape.size();
  int64_t axisIndex = concatOp.getAxis();

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
  for (unsigned dim = 0; dim < commonRank; dim++) {
    outputDims[dim] = createIE->getShapeAsDim(firstInput, dim);
  }
  IndexExpr cumulativeAxisSize = createIE->getShapeAsDim(firstInput, axisIndex);

  // Handle the rest of input
  for (unsigned i = 1; i < numInputs; ++i) {
    Value currInput = operandAdaptor.getInputs()[i];
    for (unsigned dim = 0; dim < commonRank; dim++) {
      if (dim == axisIndex) {
        IndexExpr currentSize = createIE->getShapeAsDim(currInput, axisIndex);
        cumulativeAxisSize = cumulativeAxisSize + currentSize;
      } else {
        IndexExpr possiblyLiteralDim = createIE->getShapeAsDim(currInput, dim);
        if (possiblyLiteralDim.isLiteral()) {
          // The size of current dimension of current input  is a constant
          outputDims[dim] = possiblyLiteralDim;
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
  if (!hasShapeAndRank(getOperation()))
    return success();

  ShapedType commonType;
  for (Value operand : operandAdaptor.getOperands()) {
    ArrayRef<int64_t> operandShape =
        mlir::cast<ShapedType>(operand.getType()).getShape();
    if (operandShape.size() == 1 && operandShape[0] == 0) {
      continue;
    } else {
      commonType = mlir::cast<ShapedType>(operand.getType());
      break;
    }
  }

  ArrayRef<int64_t> commonShape = commonType.getShape();
  int64_t commonRank = commonShape.size();
  int64_t axisIndex = getAxis();

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
        mlir::cast<ShapedType>(operand.getType()).getShape();
    int64_t operandRank = operandShape.size();
    // Empty tensor.
    if (operandRank == 1 && operandShape[0] == 0)
      continue;
    // Non-empty tensor.
    if (operandRank != commonRank)
      return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
          *this->getOperation(), operand, operandRank,
          std::to_string(commonRank));

    for (int64_t dim = 0; dim < commonRank; ++dim) {
      if (dim == axisIndex)
        continue;
      if (!ShapedType::isDynamic(operandShape[dim]) &&
          !ShapedType::isDynamic(commonShape[dim]) &&
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
    std::function<void(Region &)> doShapeInference) {
  // The check of constraints is kept
  // However, current check handles dynamic dim only for the concat dim
  if (!hasShapeAndRank(getOperation()))
    return success();
  // Checking value of axis parameter.
  auto commonType = mlir::cast<RankedTensorType>(getOperand(0).getType());
  auto commonShape = commonType.getShape();
  int64_t commonRank = commonShape.size();
  int64_t axisIndex = getAxis();
  // Negative axis means values are counted from the opposite side.
  if (axisIndex < 0) {
    axisIndex = commonRank + axisIndex;
    // Tong Chen:
    // TOFIX: attribute modification should be into canonicalization
    // I did not move the code into ShapeHelper
    auto builder = Builder(getContext());
    setAxisAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, /*value=*/axisIndex, /*isSigned=*/true)));
  }

  ONNXConcatOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(commonType.getElementType());
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXConcatOp>;
} // namespace onnx_mlir
