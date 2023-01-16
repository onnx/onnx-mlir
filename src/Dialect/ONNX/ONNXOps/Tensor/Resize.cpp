/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Resize.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Resize operation.
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

template <>
LogicalResult ONNXResizeOpShapeHelper::computeShape() {
  ONNXResizeOpAdaptor operandAdaptor(operands);
  Value input = operandAdaptor.X();
  uint64_t rank = createIE->getShapedTypeRank(input);
  DimsExpr outputDims;

  bool scalesFromNone = isFromNone(operandAdaptor.scales());
  if (!scalesFromNone) {
    createIE->getShapeAsDims(input, outputDims);
    DimsExpr scales;
    createIE->getIntFromArrayAsSymbols(operandAdaptor.scales(), scales);
    for (uint64_t i = 0; i < rank; ++i)
      outputDims[i] = outputDims[i] * scales[i];
  } else {
    createIE->getIntFromArrayAsSymbols(operandAdaptor.sizes(), outputDims);
  }
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXResizeOp::verify() {
  if (!hasShapeAndRank(X())) {
    return success();
  }

  bool scalesFromNone = isFromNone(scales());
  bool sizesFromNone = isFromNone(sizes());
  if (scalesFromNone == sizesFromNone) {
    if (scalesFromNone)
      return emitError("scales() and sizes() can not be both None");
    else
      return emitError("scales() and sizes() can not be both defined");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXResizeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(X()))
    return success();

  // TODO : Remove this if branch once floating point scales are handled in
  // ONNXResizeOpShapeHelper Issue number : #1958
  if (!isFromNone(scales())) {
    auto inputTy = X().getType().cast<RankedTensorType>();

    // Output should at least has the same rank as X input
    if (!getResult().getType().isa<RankedTensorType>()) {
      SmallVector<int64_t, 4> dims(inputTy.getRank(), -1);
      getResult().setType(
          RankedTensorType::get(dims, inputTy.getElementType()));
    }

    ElementsAttr scalesAttrs = getElementAttributeFromONNXValue(scales());
    if (!scalesAttrs) {
      return success();
    }

    SmallVector<float, 4> scalesConstant;
    for (auto scaleAttr : scalesAttrs.getValues<FloatAttr>()) {
      scalesConstant.emplace_back(scaleAttr.getValueAsDouble());
    }

    SmallVector<int64_t, 4> dims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      int newDim;
      if (ShapedType::isDynamic(inputTy.getShape()[i]))
        newDim = -1;
      else
        newDim = inputTy.getShape()[i] * scalesConstant[i];
      dims.emplace_back(newDim);
    }

    updateType(getResult(), dims, inputTy.getElementType());
    return success();
  }

  Type elementType = X().getType().cast<RankedTensorType>().getElementType();
  ONNXResizeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXResizeOp>;
} // namespace onnx_mlir
