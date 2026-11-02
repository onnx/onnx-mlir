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

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

LogicalResult ONNXPadOpShapeHelper::getPadAxes(
    SmallVectorImpl<IndexExpr> &axes) {
  ONNXPadOpAdaptor operandAdaptor(operands);
  Value dataOperand = operandAdaptor.getData();
  Value axesOperand = operandAdaptor.getAxes();
  axes.clear();

  // Without `axes`, all the axes of `data` are padded, in order.
  if (isNoneValue(axesOperand)) {
    if (!hasShapeAndRank(dataOperand))
      return failure();
    uint64_t dataRank = createIE->getShapedTypeRank(dataOperand);
    for (uint64_t i = 0; i < dataRank; ++i)
      axes.emplace_back(LitIE(i));
    return success();
  }

  // The number of axes must be static; their values may not be known yet,
  // e.g. when some come from a Concat that constant propagation has not
  // folded. The verifier reports an `axes` that is not 1-D.
  if (!hasShapeAndRank(axesOperand) ||
      createIE->getShapedTypeRank(axesOperand) != 1)
    return failure();
  int64_t numAxes = createIE->getArraySize(axesOperand);
  if (numAxes == ShapedType::kDynamic)
    return failure();
  for (int64_t k = 0; k < numAxes; ++k)
    axes.emplace_back(createIE->getIntFromArrayAsSymbol(axesOperand, k));
  return success();
}

LogicalResult ONNXPadOpShapeHelper::computeShape() {
  ONNXPadOpAdaptor operandAdaptor(operands);
  Value dataOperand = operandAdaptor.getData();
  Value padsOperand = operandAdaptor.getPads();
  DimsExpr outputDims;

  // Get info about input data operand.
  if (!hasShapeAndRank(dataOperand)) {
    return failure();
  }
  int64_t dataRank = createIE->getShapedTypeRank(dataOperand);

  // Get the padded axes. They may be listed in any order, and need not be
  // contiguous. Fail without a diagnostic when some are not known: shape
  // inference handles that case before calling here, and the pre-Krnl verify
  // pass reports it once all constant propagation is done.
  SmallVector<IndexExpr, 4> axes;
  if (failed(getPadAxes(axes)))
    return failure();
  int64_t numAxes = axes.size();
  // Position in `axes` of each axis of `data`, or -1 when it is not padded.
  SmallVector<int64_t, 4> axesPos(dataRank, -1);
  for (int64_t k = 0; k < numAxes; ++k) {
    if (!axes[k].isLiteral())
      return failure();
    int64_t axis = axes[k].getLiteral();
    if (axis < 0)
      axis += dataRank;
    if (axis < 0 || axis >= dataRank || axesPos[axis] != -1)
      return op->emitError("Pad axes[") << k << "] is out of range or repeated";
    axesPos[axis] = k;
  }

  // Initialize context and results (pads & output)
  pads.resize(2 * dataRank); // pads two sides of each axis.
  outputDims.resize(dataRank);

  // `pads` format is : [x1_begin, x2_begin,...,x1_end, x2_end,...],
  // where
  // - xi_begin: the number of pad values added at the beginning of axis
  //   `axes[i]`
  // - xi_end: the number of pad values added at the end of axis `axes[i]`.
  // The `pads` saved here are instead indexed by the axes of `data`, with 0
  // for axes not listed in `axes`.

  // Calculate output dimension sizes.
  for (int64_t i = 0; i < dataRank; i++) {
    // Get begin/end pads.
    IndexExpr padBegin = LitIE(0);
    IndexExpr padEnd = LitIE(0);
    if (int64_t k = axesPos[i]; k >= 0) {
      padBegin = SymIE(createIE->getIntFromArrayAsSymbol(padsOperand, k));
      padEnd =
          SymIE(createIE->getIntFromArrayAsSymbol(padsOperand, k + numAxes));
      if (padBegin.isUndefined() || padEnd.isUndefined())
        return op->emitError("pad parameter could not be processed");
    }
    // Get input dim.
    DimIndexExpr dimInput(createIE->getShapeAsDim(dataOperand, i));

    // Calculation for output size.
    IndexExpr dimOutputFinal = (padBegin + dimInput) + padEnd;

    // Save results.
    pads[i] = padBegin;
    pads[i + dataRank] = padEnd;
    outputDims[i] = dimOutputFinal;
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
  ShapedType dataTy = mlir::cast<ShapedType>(getData().getType());
  Type constTy = getConstantValue().getType();

  if (!isNoneValue(getConstantValue())) {
    // Check that the constant has the same element type as the input
    ShapedType shapedConstTy = mlir::cast<ShapedType>(constTy);
    if (dataTy.getElementType() != shapedConstTy.getElementType()) {
      return emitOpError("Pad with constant_value that doesn't match the "
                         "element type of the input.");
    }
  }

  StringRef mode = getMode();
  if (!mode.equals_insensitive("constant") &&
      !mode.equals_insensitive("reflect") && !mode.equals_insensitive("edge") &&
      !mode.equals_insensitive("wrap"))
    return emitOpError("mode must be one of constant, reflect, edge or wrap, "
                       "got ")
           << mode;

  // Test that pads/axes are 1-D tensors.
  ShapedType padsTy = mlir::cast<ShapedType>(getPads().getType());
  if (padsTy.hasRank() && padsTy.getRank() != 1)
    return emitOpError("pads must be a 1-D tensor, got rank ")
           << padsTy.getRank();
  if (!isNoneValue(getAxes())) {
    ShapedType axesTy = mlir::cast<ShapedType>(getAxes().getType());
    if (axesTy.hasRank() && axesTy.getRank() != 1)
      return emitOpError("axes must be a 1-D tensor, got rank ")
             << axesTy.getRank();
  }

  // Check the axes whose values are known at this point; the others are
  // checked once constant propagation has made them known.
  if (isNoneValue(getAxes()) || !hasShapeAndRank(getData()))
    return success();
  int64_t dataRank = dataTy.getRank();
  ONNXPadOpShapeHelper shapeHelper(getOperation(), {});
  SmallVector<IndexExpr, 4> axes;
  if (failed(shapeHelper.getPadAxes(axes)))
    return success();
  int64_t numAxes = axes.size();
  SmallVector<int64_t, 4> axesPos(dataRank, -1);
  for (int64_t k = 0; k < numAxes; ++k) {
    if (!axes[k].isLiteral())
      continue;
    int64_t axis = axes[k].getLiteral();
    if (axis < -dataRank || axis >= dataRank)
      return emitOpError("axes[") << k << "] = " << axis << " is out of range ["
                                  << -dataRank << ", " << dataRank - 1 << "]";
    if (axis < 0)
      axis += dataRank;
    if (axesPos[axis] != -1)
      return emitOpError("axes[") << k << "] repeats axis " << axis
                                  << " of axes[" << axesPos[axis] << "]";
    axesPos[axis] = k;
  }
  if (padsTy.hasStaticShape() && padsTy.getDimSize(0) != 2 * numAxes)
    return emitOpError("pads must have 2 * ")
           << numAxes << " elements, one begin and one end per padded axis, "
           << "got " << padsTy.getDimSize(0);

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

  Type elementType =
      mlir::cast<ShapedType>(getData().getType()).getElementType();

  ONNXPadOpShapeHelper shapeHelper(getOperation(), {});
  // Until all the axes are known, which axes grow is not known either; Pad
  // preserves the rank, so the result is ranked with all dims dynamic.
  SmallVector<IndexExpr, 4> axes;
  if (failed(shapeHelper.getPadAxes(axes)) ||
      llvm::any_of(axes, [](IndexExpr axis) { return !axis.isLiteral(); })) {
    int64_t dataRank = mlir::cast<ShapedType>(getData().getType()).getRank();
    SmallVector<int64_t, 4> dims(dataRank, ShapedType::kDynamic);
    updateType(getOperation(), getResult(), dims, elementType);
    return success();
  }
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
