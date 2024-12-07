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
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());
  OP_TYPE splitOp = llvm::cast<OP_TYPE>(op);

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
    for (unsigned int i = 0; i < numOfResults; ++i) {
      LiteralIndexExpr dim(indexExprArray[i]);
      splitDims.emplace_back(dim);
    }
  } else {
    // If split parameter is not specified, the dimension is split to
    // equal-sized parts.
    bool hasNumOutputsAttr = std::is_same_v<OP_TYPE, ONNXSplitOp>;
    // TODO figure out how to handle when numResults is determined by
    // num_outputs attribute introduced for Split in opset 18
    // Currently the whole graph depends on the number of outputs being
    // determined at the ONNX to ONNX-MLIR ingestion stage.
    IndexExpr splitInputDim = createIE->getShapeAsDim(input, axisIndex);
    LiteralIndexExpr numOfPartitions(numOfResults);
    if (splitInputDim.isLiteral() &&
        (splitInputDim.getLiteral() % numOfResults != 0) && !hasNumOutputsAttr)
      return op->emitError("The dimension at the split axis is "
                           "expected to be divisible by the number of results");

    unsigned numBiggerChunks = splitInputDim.isLiteral()
                                   ? splitInputDim.getLiteral() % numOfResults
                                   : numOfResults;
    for (unsigned int i = 0; i < numOfResults; ++i) {
      IndexExpr splitDim = (i < numBiggerChunks)
                               ? splitInputDim.ceilDiv(numOfPartitions)
                               : splitInputDim.floorDiv(numOfPartitions);
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
  ONNXSplitOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  Value split = operandAdaptor.getSplit();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (isNoneValue(split)) {
    // None is fine, indexExprArray will be empty.
  } else {
    createIE->getIntFromArrayAsSymbols(split, indexExprArray);
    assert(IndexExpr::isLiteral(indexExprArray) &&
           "dynamic split not yet supported");
  }
  return customComputeShape(indexExprArray);
}

// Code for SplitV13Op compute shape.
template <>
LogicalResult ONNXSplitV13OpShapeHelper::computeShape() {
  ONNXSplitOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  Value split = operandAdaptor.getSplit();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (isNoneValue(split)) {
    // None is fine, indexExprArray will be empty.
  } else {
    createIE->getIntFromArrayAsSymbols(split, indexExprArray);
    assert(IndexExpr::isLiteral(indexExprArray) &&
           "dynamic split not yet supported");
  }
  return customComputeShape(indexExprArray);
}

// Code for SplitV11Op compute shape.
template <>
LogicalResult ONNXSplitV11OpShapeHelper::computeShape() {
  ONNXSplitV11OpAdaptor operandAdaptor(operands, op->getAttrDictionary());
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
