/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Slice.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Slice operation.
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

LogicalResult ONNXSliceOpShapeHelper::computeShape() {
  // Get info about input data operand.
  ONNXSliceOpAdaptor operandAdaptor(operands);
  Value data = operandAdaptor.getData();
  uint64_t dataRank = mlir::cast<ShapedType>(data.getType()).getShape().size();

  // Get each of the axes, and save the literal values in axesIntLit.
  SmallVector<int64_t, 4> axesIntLit;
  Value axes = operandAdaptor.getAxes();
  if (isNoneValue(axes)) {
    // If `axes` are omitted, they are set to `[0, ..., nDim-1]`."
    for (uint64_t i = 0; i < dataRank; ++i)
      axesIntLit.emplace_back(i);
  } else {
    SmallVector<IndexExpr, 4> axesSymbol;
    createIE->getIntFromArrayAsSymbols(axes, axesSymbol);
    for (IndexExpr val : axesSymbol) {
      if (!val.isLiteral())
        return op->emitError("Axes must be known at compile time");
      int64_t axis = val.getLiteral();
      if (axis < 0)
        axis += dataRank;
      if (!(axis >= 0 && axis < static_cast<int64_t>(dataRank)))
        return op->emitError("Axes contains an out-of-bound index");
      axesIntLit.emplace_back(axis);
    }
  }
  uint64_t sliceRank = axesIntLit.size();

  // Initialize context and results (start & output)
  starts.resize(dataRank);
  steps.resize(dataRank);
  ends.resize(dataRank);
  DimsExpr outputDims;
  outputDims.resize(dataRank);

  for (uint64_t i = 0; i < sliceRank; i++) {
    // i is index in start/step/end/output
    // ii is logical index in mem/loop bounds
    int ii = axesIntLit[i];
    // Get start, end, step, and dim index expressions.
    // Get start.
    SymbolIndexExpr startInput =
        createIE->getIntFromArrayAsSymbol(operandAdaptor.getStarts(), i);
    if (startInput.isUndefined())
      return op->emitError("start input parameter could not be processed");
    // Get end.
    SymbolIndexExpr endInput =
        createIE->getIntFromArrayAsSymbol(operandAdaptor.getEnds(), i);
    if (endInput.isUndefined())
      return op->emitError("end input parameter could not be processed");
    // Get step.
    SymbolIndexExpr stepInput =
        createIE->getIntFromArrayAsSymbol(operandAdaptor.getSteps(), i);
    if (stepInput.isUndefined())
      return op->emitError("step input parameter could not be processed");
    if (stepInput.isLiteral() && stepInput.getLiteral() == 0)
      return op->emitError("step input parameter cannot be zero");
    // Get dim.
    DimIndexExpr dimInput = createIE->getShapeAsDim(data, ii);

    // Now proceed with the computations for start/end/dim.
    // Calculation for start: start < 0 ? start + dim : start.
    IndexExpr startPos =
        IndexExpr::select(startInput < 0, startInput + dimInput, startInput);
    // Step < 0: clamp(0, start, dim -1) else clamp(0, start, dim)
    IndexExpr neg = startPos.clamp(0, dimInput - 1);
    IndexExpr pos = startPos.clamp(0, dimInput);
    IndexExpr startFinal = IndexExpr::select(stepInput < 0, neg, pos);

    // Calculation for end: end<0 -> end + dim else -> end;
    // special case end <= -inf -> -1;  end >= inf -> dim;
    int64_t negInf = std::numeric_limits<int32_t>::min();
    int64_t posInf = std::numeric_limits<int32_t>::max();
    IndexExpr endPos =
        IndexExpr::select(endInput < 0, endInput + dimInput, endInput);
    endPos = endPos.selectOrSelf(endInput <= negInf, -1);
    endPos = endPos.selectOrSelf(endInput >= posInf, dimInput);
    // End: step<0: clamp(-1, end, dim); step>0 clamp(0, end, dim)
    neg = endPos.clamp(-1, dimInput);
    pos = endPos.clamp(0, dimInput);
    IndexExpr endFinal = IndexExpr::select(stepInput < 0, neg, pos);

    // Calculation for output size.
    IndexExpr dimOutputFinal = (endFinal - startFinal).ceilDiv(stepInput);
    // should use a max
    dimOutputFinal = dimOutputFinal.selectOrSelf(dimOutputFinal < 0, 0);

    // Save results
    starts[ii] = startFinal;
    steps[ii] = stepInput;
    ends[ii] = endFinal;
    outputDims[ii] = dimOutputFinal;
  }

  // Handle the default for the non-axis arrays; they are detected with 0
  // steps (illegal value).
  for (uint64_t i = 0; i < dataRank; ++i) {
    if (steps[i].isUndefined()) {
      // have one unset, put the defaults (start was already at zero, so we
      // are fine).
      starts[i] = LitIE(0);
      steps[i] = LitIE(1);
      DimIndexExpr dimInput = createIE->getShapeAsDim(data, i);
      ends[i] = dimInput;
      outputDims[i] = dimInput;
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

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXSliceOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getData()))
    return success();

  Value axes = getAxes();
  Value steps = getSteps();

  // Cannot infer shape if axes is not a constant. It can be a constant after
  // several rounds of shape-inference and constant propagation.
  if (!isNoneValue(axes) && !getONNXConstantOp(axes))
    return success();

  const auto startsType =
      mlir::dyn_cast<RankedTensorType>(getStarts().getType());
  assert(startsType != nullptr && "starts type is not a RankedTensorType");
  auto startsDim = startsType.getShape()[0];
  {
    OpBuilder builder(this->getContext());
    OnnxBuilder createONNX(builder, this->getLoc());
    const Type elementType = builder.getIntegerType(64);
    const auto tensorType = RankedTensorType::get({startsDim}, elementType);

    // If axes is not specified, default to [0, ..., ndim-1]
    if (isNoneValue(axes)) {
      SmallVector<int64_t, 1> vals = {};
      for (size_t s = 0; s < static_cast<size_t>(startsDim); ++s)
        vals.emplace_back(s);
      auto constantDenseAttribute =
          DenseElementsAttr::get(tensorType, llvm::ArrayRef(vals));
      builder.setInsertionPoint(*this);
      Value constantResult = createONNX.constant(constantDenseAttribute);
      this->setOperand(3, constantResult);
    }

    // If steps is not specified, default to [1, ..., 1]
    if (isNoneValue(steps)) {
      SmallVector<int64_t, 1> vals(startsDim, 1);
      auto constantDenseAttribute =
          DenseElementsAttr::get(tensorType, llvm::ArrayRef(vals));
      builder.setInsertionPoint(*this);
      Value constantResult = createONNX.constant(constantDenseAttribute);
      this->setOperand(4, constantResult);
    }
  }

  Type elementType =
      mlir::cast<ShapedType>(getData().getType()).getElementType();
  ONNXSliceOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
