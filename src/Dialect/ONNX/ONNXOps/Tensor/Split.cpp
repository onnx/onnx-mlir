/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Split.cpp - ONNX Operations -----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Split operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {


#if 1
template <typename ShapeHelper, typename OperandAdaptor>
LogicalResult ONNXSplitOpShapeHelperCommon(ShapeHelper *shapeHelper,
    OperandAdaptor operandAdaptor, ArrayRef<IndexExpr> indexExprArray) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Get info about input and output data.
  auto op = shapeHelper->op;
  unsigned int numOfResults = op->getNumResults();
  auto rank =
      operandAdaptor.input().getType().template cast<ShapedType>().getRank();

  // Checking value of axis parameter.
  int64_t axisIndex = op->axis();
  if (axisIndex < -rank || axisIndex >= rank)
    return op->emitError("Split axis value out of bound");
  // Negative axis means values are counted from the opposite side.
  if (axisIndex < 0) {
    axisIndex = rank + axisIndex;
    auto builder = mlir::Builder(op->getContext());
    op->axisAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, /*value=*/axisIndex, /*isSigned=*/true)));
  }

  SmallVector<IndexExpr, 4> splitDims;
  MemRefBoundsIndexCapture inputBounds(operandAdaptor.input());
  if (!indexExprArray.empty()) {
    if (indexExprArray.size() != numOfResults)
      return op->emitError("Split size not equal to the number of results");
    for (unsigned int i = 0; i < numOfResults; ++i) {
      LiteralIndexExpr dim(indexExprArray[i]);
      splitDims.emplace_back(dim);
    }
  } else {
    // If split parameter is not specified, the dimension is split to
    // equal-sized parts.
    DimIndexExpr splitInputDim(inputBounds.getDim(axisIndex));
    LiteralIndexExpr numOfPartitions(numOfResults);
    if (splitInputDim.isLiteral() &&
        (splitInputDim.getLiteral() % numOfResults != 0))
      return op->emitError("The dimension at the split axis is "
                           "expected to be divisible by the number of results");
    for (unsigned int i = 0; i < numOfResults; ++i) {
      IndexExpr splitDim = splitInputDim.ceilDiv(numOfPartitions);
      splitDims.emplace_back(splitDim);
    }
  }

  // Build result types.
  for (unsigned int i = 0; i < numOfResults; ++i) {
    DimsExpr outputDims;
    outputDims.resize(rank);
    for (unsigned int j = 0; j < rank; ++j) {
      if (j == axisIndex) {
        outputDims[j] = splitDims[i];
      } else {
        outputDims[j] = inputBounds.getDim(j);
      }
    }
    shapeHelper->dimsForOutput(i) = outputDims;
  }
  return success();
}

LogicalResult ONNXSplitOpShapeHelper::computeShape(
    ONNXSplitOpAdaptor operandAdaptor) {

  auto split = op->split();

  SmallVector<IndexExpr, 4> indexExprArray;
  // TODO: getONNXConstantOp might be a problem during code gen as ONNX
  // constant get lowered to global constants.
  if (auto splitConstOp = getONNXConstantOp(split)) {
    ArrayValueIndexCapture splitCapture(split, fGetDenseVal, fLoadVal);
    auto splitRank =
        splitConstOp.valueAttr().dyn_cast_or_null<DenseElementsAttr>().size();
    splitCapture.getSymbolList(splitRank, indexExprArray);
  } else if (!split.getType().template isa<NoneType>()) {
    llvm_unreachable("dynamic split not yet supported");
  }

  return ONNXSplitOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

LogicalResult ONNXSplitV11OpShapeHelper::computeShape(
    ONNXSplitV11OpAdaptor operandAdaptor) {
  auto splitAttr = op->split();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (splitAttr.has_value()) {
    ArrayAttributeIndexCapture splitCapture(splitAttr.value());
    auto splitRank = splitCapture.size();
    for (unsigned i = 0; i < splitRank; ++i) {
      indexExprArray.emplace_back(splitCapture.getLiteral(i));
    }
  }
  return ONNXSplitOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXSplitOp::verify() {
  ONNXSplitOpAdaptor operandAdaptor(*this);
  Value input = operandAdaptor.input();
  if (!hasShapeAndRank(input))
    return success(); // Won't be able to do any checking at this stage.

  auto inputType = input.getType().cast<ShapedType>();
  int64_t inputRank = inputType.getShape().size();
  int64_t axisIndex = axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(input).
  if (axisIndex < -inputRank || axisIndex >= inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXSplitOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape isn't known yet.
  if (!hasShapeAndRank(input()))
    return success();

  auto inputType = input().getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  SmallVector<Type> elementTypes(getNumResults(), elementType);

  return shapeHelperInferMultipleShapes<ONNXSplitOpShapeHelper, ONNXSplitOp,
      ONNXSplitOpAdaptor>(*this, elementTypes);
}

LogicalResult ONNXSplitV11Op::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape isn't known yet.
  if (!hasShapeAndRank(input()))
    return success();

  auto inputType = input().getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  SmallVector<Type> elementTypes(getNumResults(), elementType);

  return shapeHelperInferMultipleShapes<ONNXSplitV11OpShapeHelper,
      ONNXSplitV11Op, ONNXSplitV11OpAdaptor>(*this, elementTypes);
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//
