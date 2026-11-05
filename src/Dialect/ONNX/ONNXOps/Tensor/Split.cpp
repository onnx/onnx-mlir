/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Split.cpp - ONNX Operations -----------------------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Split operation.
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

// Code common for all split ops.
template <typename OP_TYPE>
LogicalResult ONNXCommonSplitOpShapeHelper<OP_TYPE>::customComputeShape(
    ArrayRef<IndexExpr> indexExprArray) {
  OP_TYPE splitOp = mlir::dyn_cast<OP_TYPE>(op);
  typename OP_TYPE::Adaptor operandAdaptor(operands, splitOp);

  unsigned int numOfResults = splitOp.getNumResults();
  Value input = operandAdaptor.getInput();
  if (!hasShapeAndRank(input)) {
    return failure();
  }
  int64_t rank = createIE->getShapedTypeRank(input);

  // Checking value of axis parameter.
  int64_t axisIndex = operandAdaptor.getAxis();
  if (axisIndex < -rank || axisIndex >= rank)
    return op->emitError("Split axis value out of bound");
  // Negative axis means values are counted from the opposite side.
  if (axisIndex < 0) {
    axisIndex = rank + axisIndex;
    auto builder = Builder(op->getContext());
    splitOp.setAxisAttr(
        IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
            APInt(64, /*value=*/axisIndex, /*isSigned=*/true)));
  }

  SmallVector<IndexExpr, 4> splitDims;
  if (!indexExprArray.empty()) {
    if (indexExprArray.size() != numOfResults)
      return op->emitError("Split size not equal to the number of results");
    IndexExpr splitInputDim = createIE->getShapeAsDim(input, axisIndex);
    int64_t numOfNonLiterals = 0, sumOfLiterals = 0, nonLiteralIndex = -1;
    for (unsigned int i = 0; i < numOfResults; ++i) {
      if (indexExprArray[i].isLiteral()) {
        sumOfLiterals += indexExprArray[i].getLiteral();
      } else {
        numOfNonLiterals++;
        nonLiteralIndex = i;
      }
      splitDims.emplace_back(indexExprArray[i]);
    }
    // When the size of the split dimension is known, check that the sum of the
    // split sizes is equal to the size of the split dimension.
    if (splitInputDim.isLiteral()) {
      int64_t axisDim = splitInputDim.getLiteral();
      if (numOfNonLiterals == 0 && sumOfLiterals != axisDim)
        return op->emitError("Split sizes must sum up to the dimension at "
                             "the split axis");
      // A single unknown split value is the remainder of the axis dimension.
      if (numOfNonLiterals == 1 && sumOfLiterals <= axisDim)
        splitDims[nonLiteralIndex] = LitIE(axisDim - sumOfLiterals);
    }
  } else {
    IndexExpr splitInputDim = createIE->getShapeAsDim(input, axisIndex);
    LiteralIndexExpr numOfPartitions(numOfResults);

    // SplitV13-to-Split decomposition currently represents an absent split
    // using ONNXSplitOp without num_outputs. Preserve equal-split semantics
    // for that compatibility form and use Split-18 semantics only when the
    // num_outputs attribute is actually present.
    bool hasNumOutputs = false;
    if constexpr (std::is_same_v<OP_TYPE, ONNXSplitOp>)
      hasNumOutputs = splitOp.getNumOutputs().has_value();

    if (hasNumOutputs) {
      // Split-18: use ceil-sized chunks, except for the last chunk,
      // which contains all remaining elements.
      for (unsigned i = 0; i + 1 < numOfResults; ++i)
        splitDims.emplace_back(splitInputDim.ceilDiv(numOfPartitions));

      // Build a distinct expression for each result. Downstream IndexExpr
      // lowering classifies these values independently as dimensions or
      // symbols, so sharing one computed IndexExpr between results is unsafe.
      IndexExpr lastChunk =
          splitInputDim - splitInputDim.ceilDiv(numOfPartitions) *
                              LiteralIndexExpr(numOfResults - 1);
      if (lastChunk.isLiteral() && lastChunk.getLiteral() < 0)
        return op->emitError(
            "'num_outputs' produces a negative size for the last output");
      splitDims.emplace_back(lastChunk);
    } else {
      // SplitV13, SplitV11, and the compatibility form described above must
      // produce equal-sized outputs when explicit split sizes are absent.
      if (splitInputDim.isLiteral() &&
          splitInputDim.getLiteral() % numOfResults != 0)
        return op->emitError(
            "The dimension at the split axis is expected to be "
            "divisible by the number of results");

      // For a valid dynamic input the dimension is divisible, making ceilDiv
      // equivalent to floorDiv. Keep ceilDiv to preserve the IndexExpr form
      // expected by existing lowerings.
      for (unsigned i = 0; i < numOfResults; ++i)
        splitDims.emplace_back(splitInputDim.ceilDiv(numOfPartitions));
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
        outputDims[j] = createIE->getShapeAsDim(input, j);
      }
    }
    setOutputDims(outputDims, i);
  }
  return success();
}

// Code for SplitOp compute shape.
template <>
LogicalResult ONNXSplitOpShapeHelper::computeShape() {
  ONNXSplitOpAdaptor operandAdaptor(operands, mlir::dyn_cast<ONNXSplitOp>(op));
  Value split = operandAdaptor.getSplit();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (isNoneValue(split)) {
    // None is fine, indexExprArray will be empty.
  } else {
    // Split values may be runtime values; they are then question marks
    // during analysis and symbols during code generation.
    createIE->getIntFromArrayAsSymbols(split, indexExprArray);
  }
  return customComputeShape(indexExprArray);
}

// Code for SplitV13Op compute shape.
template <>
LogicalResult ONNXSplitV13OpShapeHelper::computeShape() {
  ONNXSplitV13OpAdaptor operandAdaptor(
      operands, mlir::dyn_cast<ONNXSplitV13Op>(op));
  Value split = operandAdaptor.getSplit();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (isNoneValue(split)) {
    // None is fine, indexExprArray will be empty.
  } else {
    // Split values may be runtime values; they are then question marks
    // during analysis and symbols during code generation.
    createIE->getIntFromArrayAsSymbols(split, indexExprArray);
  }
  return customComputeShape(indexExprArray);
}

// Code for SplitV11Op compute shape.
template <>
LogicalResult ONNXSplitV11OpShapeHelper::computeShape() {
  ONNXSplitV11OpAdaptor operandAdaptor(
      operands, mlir::dyn_cast<ONNXSplitV11Op>(op));
  ArrayAttr splitAttr = operandAdaptor.getSplitAttr();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (splitAttr) {
    createIE->getIntFromArrayAsLiterals(splitAttr, indexExprArray);
  }
  return customComputeShape(indexExprArray);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXSplitOp::verify() {
  ONNXSplitOpAdaptor operandAdaptor(*this);
  Value input = operandAdaptor.getInput();
  Value split = operandAdaptor.getSplit();
  std::optional<int64_t> numOutputs = operandAdaptor.getNumOutputs();

  if (getNumResults() == 0)
    return emitOpError("requires at least one output");

  if (!isNoneValue(split) && numOutputs.has_value())
    return emitOpError(
        "does not allow both 'split' and 'num_outputs' to be specified");

  if (numOutputs.has_value()) {
    if (*numOutputs < 1)
      return emitOpError("requires 'num_outputs' to be positive");
    if (static_cast<uint64_t>(*numOutputs) != getNumResults())
      return emitOpError("requires 'num_outputs' to equal the number of "
                         "variadic results");
  }

  if (!hasShapeAndRank(input))
    return success(); // Won't be able to do any checking at this stage.

  auto inputType = mlir::cast<ShapedType>(input.getType());
  int64_t inputRank = inputType.getShape().size();
  int64_t axisIndex = getAxis();

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
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape isn't known yet.
  if (!hasShapeAndRank(getInput()))
    return success();

  auto inputType = mlir::cast<ShapedType>(getInput().getType());
  Type elementType = inputType.getElementType();
  ONNXSplitOpShapeHelper shapeHelper(getOperation(), {});
  // Same time for all results.
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

LogicalResult ONNXSplitV13Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape isn't known yet.
  if (!hasShapeAndRank(getInput()))
    return success();

  auto inputType = mlir::cast<ShapedType>(getInput().getType());
  Type elementType = inputType.getElementType();
  ONNXSplitV13OpShapeHelper shapeHelper(getOperation(), {});
  // Same time for all results.
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

LogicalResult ONNXSplitV11Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape isn't known yet.
  if (!hasShapeAndRank(getInput()))
    return success();

  auto inputType = mlir::cast<ShapedType>(getInput().getType());
  Type elementType = inputType.getElementType();
  ONNXSplitV11OpShapeHelper shapeHelper(getOperation(), {});
  // Same time for all results.
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXCommonSplitOpShapeHelper<ONNXSplitOp>;
template struct ONNXCommonSplitOpShapeHelper<ONNXSplitV13Op>;
template struct ONNXCommonSplitOpShapeHelper<ONNXSplitV11Op>;
} // namespace onnx_mlir
