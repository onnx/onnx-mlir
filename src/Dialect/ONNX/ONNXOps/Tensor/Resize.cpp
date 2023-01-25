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

LogicalResult ONNXResizeOpShapeHelper::computeShape() {
  ONNXResizeOpAdaptor operandAdaptor(operands);
  uint64_t rank = createIE->getShapedTypeRank(operandAdaptor.X());
  DimsExpr inputDims, outputDims;
  createIE->getShapeAsDims(operandAdaptor.X(), inputDims);
  bool scalesFromNone = isFromNone(operandAdaptor.scales());

  if (!scalesFromNone) {
    // Read and save scales as float.
    createIE->getFloatFromArrayAsNonAffine(operandAdaptor.scales(), scales);
    if (inputDims.size() != scales.size())
      return op->emitError("expected scales to have the same rank as input");
    // Compute output dims = int(floor(float(input dims) * scales)).
    for (uint64_t i = 0; i < rank; ++i) {
      // Special case for scale == 1.0 as converts are then needed.
      if (scales[i].isLiteralAndIdenticalTo(1.0)) {
        outputDims.emplace_back(inputDims[i]);
      } else {
        IndexExpr floatInputDim = inputDims[i].convertToFloat();
        // hi alex
        inputDims[i].debugPrint("input dims as int");
        floatInputDim.debugPrint("input dims as float");
        scales[i].debugPrint("scales as float");
        IndexExpr floatProduct = floatInputDim * scales[i];
        // Formula has a floor, but convert of positive number already rounds
        // toward zero, so skip the floor. 
        outputDims.emplace_back(floatProduct.convertToIndex());
      }
    }
  } else {
    // Output size is defined by input `sizes`.
    createIE->getIntFromArrayAsSymbols(operandAdaptor.sizes(), outputDims);
    if (inputDims.size() != outputDims.size())
      return op->emitError("expected scales to have the same rank as input");
    // Compute scales as float(output dims) / float(input dims).
    for (uint64_t i = 0; i < rank; ++i) {
      IndexExpr floatInputDim = inputDims[i].convertToFloat();
      IndexExpr floatOutputDim = outputDims[i].convertToFloat();
      scales.emplace_back(floatOutputDim / floatInputDim);
    }
  }
  // Save output dims
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
  // Should test the sizes of scales or size to be the same as the rank of X.
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXResizeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(X()))
    return success();

#if 0 // hi alex
  // TODO : Remove this if branch once floating point scales are handled in
  // ONNXResizeOpShapeHelper Issue number : #1958
  if (!isFromNone(scales())) {
    RankedTensorType inputTy = X().getType().cast<RankedTensorType>();

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
#endif

  Type elementType = X().getType().cast<RankedTensorType>().getElementType();
  ONNXResizeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
