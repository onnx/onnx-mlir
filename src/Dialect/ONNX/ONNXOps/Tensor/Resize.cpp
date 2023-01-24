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
#if 1 // hi alex
    DimsExpr inputDims;
    createIE->getShapeAsDims(input, inputDims);
    DimsExpr floatScales;
    createIE->getFloatFromArrayAsNonAffine(
        operandAdaptor.scales(), floatScales);
    if (inputDims.size() != floatScales.size())
      return op->emitError("expected scales to have the same rank as input");
    for (uint64_t i = 0; i < rank; ++i) {
      // Maybe use special case for scale == 1.0 as no floor are then needed.
      IndexExpr floatInputDim = inputDims[i].convertToFloat();
      // hi alex
      inputDims[i].debugPrint("input dims as int");
      floatInputDim.debugPrint("input dims as float");
      floatScales[i].debugPrint("scales as float");
      IndexExpr floatProduct = floatInputDim * floatScales[i];
      fprintf(stderr, "hi alex before floor\n");
      IndexExpr floatFloor = floatProduct.floor();
      floatFloor.debugPrint("hi alex after floor");
      outputDims.emplace_back(floatFloor.convertToIndex());
    }
#else
    // Old code that does not use float.
    createIE->getShapeAsDims(input, outputDims);
    DimsExpr scales;
    createIE->getIntFromArrayAsSymbols(operandAdaptor.scales(), scales);
    for (uint64_t i = 0; i < rank; ++i)
      outputDims[i] = outputDims[i] * scales[i];
#endif
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

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXResizeOp>;
} // namespace onnx_mlir
