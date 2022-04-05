/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Split.cpp - Shape Inference for Split Op ---------------===//
//
// This file implements shape inference for the ONNX Split Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

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

ONNXSplitOpShapeHelper::ONNXSplitOpShapeHelper(ONNXSplitOp *newOp)
    : ONNXOpShapeHelper<ONNXSplitOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXSplitOpShapeHelper::ONNXSplitOpShapeHelper(ONNXSplitOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXSplitOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

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

ONNXSplitV11OpShapeHelper::ONNXSplitV11OpShapeHelper(ONNXSplitV11Op *newOp)
    : ONNXOpShapeHelper<ONNXSplitV11Op>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXSplitV11OpShapeHelper::ONNXSplitV11OpShapeHelper(ONNXSplitV11Op *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXSplitV11Op>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXSplitV11OpShapeHelper::computeShape(
    ONNXSplitV11OpAdaptor operandAdaptor) {
  auto splitAttr = op->split();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (splitAttr.hasValue()) {
    ArrayAttributeIndexCapture splitCapture(splitAttr.getValue());
    auto splitRank = splitCapture.size();
    for (unsigned i = 0; i < splitRank; ++i) {
      indexExprArray.emplace_back(splitCapture.getLiteral(i));
    }
  }
  return ONNXSplitOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

} // namespace onnx_mlir
