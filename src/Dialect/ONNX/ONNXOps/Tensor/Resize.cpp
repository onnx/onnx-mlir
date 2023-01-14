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
  if (!X().getType().isa<RankedTensorType>()) {
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
  if (!X().getType().isa<RankedTensorType>()) {
    return success();
  }
  // TODO : Return to this implementation once floating point scales are handled
  //
  // Type elementType = X().getType().cast<RankedTensorType>().getElementType();
  // ONNXResizeOpShapeHelper shapeHelper(getOperation(), {});
  // return shapeHelper.computeShapeAndUpdateType(elementType);

  auto inputTy = X().getType().cast<RankedTensorType>();

  // Output should at least has the same rank as X input
  if (!getResult().getType().isa<RankedTensorType>()) {
    SmallVector<int64_t, 4> dims(inputTy.getRank(), -1);
    getResult().setType(RankedTensorType::get(dims, inputTy.getElementType()));
  }

  assert(isFromNone(scales()) != isFromNone(sizes()) &&
         "Exactly one of scales and sizes can be defined");

  // Current implementation handles constant scales only
  if (!isFromNone(scales())) {
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
  } else {
    ElementsAttr sizesAttrs = getElementAttributeFromONNXValue(sizes());
    if (!sizesAttrs) {
      return success();
    }

    SmallVector<int64_t, 4> sizesConstant;
    for (auto sizeAttr : sizesAttrs.getValues<IntegerAttr>()) {
      sizesConstant.emplace_back(sizeAttr.getInt());
    }

    updateType(getResult(), sizesConstant, inputTy.getElementType());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXResizeOp>;
} // namespace onnx_mlir
