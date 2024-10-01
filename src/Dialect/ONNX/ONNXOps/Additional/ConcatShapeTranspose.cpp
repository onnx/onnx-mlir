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

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

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
LogicalResult ONNXConcatShapeTransposeOpShapeHelper::computeShape() {
  ONNXConcatShapeTransposeOpAdaptor operandAdaptor(operands);
  ONNXConcatShapeTransposeOp concatOp =
      llvm::cast<ONNXConcatShapeTransposeOp>(op);
  unsigned numInputs = concatOp.getNumOperands();
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
  DimsExpr outputConcatDims(commonRank);
  for (unsigned dim = 0; dim < commonRank; dim++) {
    outputConcatDims[dim] = createIE->getShapeAsDim(firstInput, dim);
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
          outputConcatDims[dim] = possiblyLiteralDim;
        }
      }
    }
  }
  // Dims for Concat result
  outputConcatDims[axisIndex] = cumulativeAxisSize;

  // Compute dims for ShapeOp
  Value data = operandAdaptor.getInputs()[0];
  int64_t rank = createIE->getShapedTypeRank(data);

  // Compute the normalized start/end. Negative value means counting
  // dimensions from the back.
  int64_t start = concatOp.getStart();
  int64_t end = rank;
  if (concatOp.getEnd().has_value()) {
    end = concatOp.getEnd().value();
  }
  start = normalizeClampedPerSpec(start, rank);
  end = normalizeClampedPerSpec(end, rank);
  assert(start <= end && "Start must not be greater than end");

  // Output is the actual number of values (1D)
  setOutputDims({LitIE(end - start)}, 0);

  // For the transpose
  DimsExpr outputTransposeDims(commonRank);
  ArrayAttr permAttr = concatOp.getPermAttr();
  if (!permAttr) {
    // Generate reverse order for default transpose operation.
    SmallVector<int64_t, 4> defaultVals;
    auto builder = Builder(concatOp.getContext());
    for (int i = rank - 1; i >= 0; --i)
      defaultVals.emplace_back(i);
    // Set default attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    concatOp.setPermAttr(builder.getI64ArrayAttr(defaultRefs));
    permAttr = concatOp.getPermAttr();
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

LogicalResult ONNXConcatShapeTransposeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // If any input is not ranked tensor, do nothing.
  if (!hasShapeAndRank(getOperation()))
    return success();
  auto commonType = mlir::cast<RankedTensorType>(getOperand(0).getType());
  Type intType = mlir::cast<Type>(IntegerType::get(getContext(), 64));
  SmallVector<Type> elementTypes = {intType, commonType.getElementType()};
  ONNXConcatShapeTransposeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateTypes(elementTypes);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXConcatShapeTransposeOp>;
} // namespace onnx_mlir
