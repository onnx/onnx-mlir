/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Dim.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Dim operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

// Code copied from ConcatOp, ShapeOp and TransposeOp
namespace {

// The Shape op spec says:
//
// "Note that axes will be clipped to the range [0, r-1], where r is the
// rank of the input tensor if they are out-of-range (after adding r in the case
// of negative axis). Thus, specifying any end value > r is equivalent to
// specifying an end value of r, and specifying any start value < -r is
// equivalent to specifying a start value of 0."
int64_t normalizeClampedPerSpec(int64_t axis, int64_t rank) {
  if (axis < 0)
    axis += rank;
  if (axis < 0)
    axis = 0;
  if (axis > rank)
    axis = rank;
  return axis;
}

} // namespace

namespace onnx_mlir {

template <>
LogicalResult NewONNXConcatShapeTransposeOpShapeHelper::computeShape() {
  ONNXConcatShapeTransposeOpAdaptor operandAdaptor(operands);
  ONNXConcatShapeTransposeOp concatOp =
      llvm::cast<ONNXConcatShapeTransposeOp>(op);
  unsigned numInputs = concatOp.getNumOperands();
  Value firstInput = operandAdaptor.inputs().front();
  ArrayRef<int64_t> commonShape =
      firstInput.getType().cast<ShapedType>().getShape();
  int64_t commonRank = commonShape.size();
  int64_t axisIndex = concatOp.axis();

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
  DimsExpr outputConcatDims(commonRank);
  for (unsigned dim = 0; dim < commonRank; dim++) {
    outputConcatDims[dim] = createIE->getShapeAsDim(firstInput, dim);
  }
  IndexExpr cumulativeAxisSize = createIE->getShapeAsDim(firstInput, axisIndex);

  // Handle the rest of input
  for (unsigned i = 1; i < numInputs; ++i) {
    Value currInput = operandAdaptor.inputs()[i];
    for (unsigned dim = 0; dim < commonRank; dim++) {
      if (dim == axisIndex) {
        IndexExpr currentSize = createIE->getShapeAsDim(currInput, axisIndex);
        cumulativeAxisSize = cumulativeAxisSize + currentSize;
      } else {
        IndexExpr possiblyLiteralDim = createIE->getShapeAsDim(currInput, dim);
        if (possiblyLiteralDim.isLiteral()) {
          // The size of current dimension of current input  is a constant
          outputConcatDims[dim] = possiblyLiteralDim;
        }
      }
    }
  }
  // Dims for Concat result
  outputConcatDims[axisIndex] = cumulativeAxisSize;

  // Compute dims for ShapeOp
  Value data = operandAdaptor.inputs()[0];
  // MemRefBoundsIndexCapture dataBounds(data);
  int64_t rank = createIE->getTypeRank(data);

  // Compute the normalized start/end. Negative value means counting
  // dimensions from the back.
  int64_t start = concatOp.start();
  int64_t end = rank;
  if (concatOp.end().has_value()) {
    end = concatOp.end().value();
  }
  start = normalizeClampedPerSpec(start, rank);
  end = normalizeClampedPerSpec(end, rank);
  assert(start <= end && "Start must not be greater than end");

  // Output is the actual number of values (1D)
  setOutputDims({LiteralIndexExpr(end - start)}, 0);

  // For the transpose
  DimsExpr outputTransposeDims(commonRank);
  ArrayAttr permAttr = concatOp.permAttr();
  if (!permAttr) {
    // Generate reverse order for default transpose operation.
    SmallVector<int64_t, 4> defaultVals;
    auto builder = mlir::Builder(concatOp.getContext());
    for (int i = rank - 1; i >= 0; --i)
      defaultVals.emplace_back(i);
    // Set default attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    concatOp.permAttr(builder.getI64ArrayAttr(defaultRefs));
    permAttr = concatOp.permAttr();
  }

  for (int64_t i = 0; i < commonRank; i++) {
    auto current = outputConcatDims[ArrayAttrIntVal(permAttr, i)];
    outputTransposeDims[i] = current;
  }

  setOutputDims(outputTransposeDims, 1);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

#if 1 // hi alex remove

// Compute a slice of the input tensor's shape. The slice starts from axis 0.
// The axes up to the last one will be included. Negative axes indicate counting
// back from the last axis.
std::pair<int64_t, int64_t> myDataShapeBounds_xxx(
    ONNXConcatShapeTransposeOpAdaptor &operandAdaptor) {
  Value data = operandAdaptor.inputs()[0];
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t rank = dataBounds.getRank();

  // Compute the normalized start/end. Negative value means counting
  // dimensions from the back.
  int64_t start = operandAdaptor.start();
  int64_t end = rank;
  if (operandAdaptor.end().has_value()) {
    end = operandAdaptor.end().value();
  }

  return std::make_pair(
      normalizeClampedPerSpec(start, rank), normalizeClampedPerSpec(end, rank));
}

LogicalResult ONNXConcatShapeTransposeOpShapeHelper::computeShape(
    ONNXConcatShapeTransposeOpAdaptor operandAdaptor) {
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
  DimsExpr outputConcatDims(commonRank);
  MemRefBoundsIndexCapture firstInputBounds(operandAdaptor.inputs()[0]);
  for (unsigned dim = 0; dim < commonRank; dim++) {
    outputConcatDims[dim] = firstInputBounds.getDim(dim);
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
          outputConcatDims[dim] = currInputBounds.getDim(dim);
        }
      }
    }
  }
  // Dims for Concat result
  outputConcatDims[axisIndex] = cumulativeAxisSize;

  // Compute dims for ShapeOp
  Value data = operandAdaptor.inputs()[0];
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t rank = dataBounds.getRank();

  // Compute the normalized start/end. Negative value means counting
  // dimensions from the back.
  int64_t start = operandAdaptor.start();
  int64_t end = rank;
  if (operandAdaptor.end().has_value()) {
    end = operandAdaptor.end().value();
  }

  std::tie(start, end) = myDataShapeBounds_xxx(operandAdaptor);

  assert(start <= end && "Start must not be greater than end");

  // Output is the actual number of values (1D)
  setOutputDims({LiteralIndexExpr(end - start)}, 0);

  // For the transpose
  DimsExpr outputTransposeDims(commonRank);
  ArrayAttr permAttr = operandAdaptor.permAttr();
  if (!permAttr) {
    // Generate reverse order for default transpose operation.
    SmallVector<int64_t, 4> defaultVals;
    auto builder = mlir::Builder(op->getContext());
    for (int i = rank - 1; i >= 0; --i)
      defaultVals.emplace_back(i);
    // Set default attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    op->permAttr(builder.getI64ArrayAttr(defaultRefs));
    permAttr = op->permAttr();
  }

  for (int64_t i = 0; i < commonRank; i++) {
    auto current = outputConcatDims[ArrayAttrIntVal(permAttr, i)];
    outputTransposeDims[i] = current;
  }

  setOutputDims(outputTransposeDims, 1);

  return success();
}

#endif

LogicalResult ONNXConcatShapeTransposeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {

  // If any input is not ranked tensor, do nothing.
  int inputNum = getNumOperands();
  for (int i = 0; i < inputNum; ++i) {
    if (!getOperand(i).getType().isa<RankedTensorType>())
      return success();
  }
  auto commonType = getOperand(0).getType().cast<RankedTensorType>();
  Type intType = IntegerType::get(getContext(), 64).cast<Type>();
  SmallVector<Type> elementTypes = {intType, commonType.getElementType()};
  NewONNXConcatShapeTransposeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateTypes(elementTypes);
#if 0
  return shapeHelperInferMultipleShapes<ONNXConcatShapeTransposeOpShapeHelper,
      ONNXConcatShapeTransposeOp, ONNXConcatShapeTransposeOpAdaptor>(
      *this, elementTypes);
#endif
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct NewONNXNonSpecificOpShapeHelper<ONNXConcatShapeTransposeOp>;
} // namespace onnx_mlir
