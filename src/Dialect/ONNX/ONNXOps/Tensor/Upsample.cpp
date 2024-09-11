/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Upsample.cpp - ONNX Operations --------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Upsample operation
// (deprecated).
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXUpsampleOpShapeHelper::computeShape() {
  // Read data and indices shapes as dim indices.
  ONNXUpsampleOpAdaptor operandAdaptor(operands);

  // No need to come up with a solution that generate runtime bounds as this op
  // will be converted to a resize op. If that were not the case, we would need
  // to load the float values, convert the shapes to float, multiply, reconvert
  // back to int. No need for that at this moment.
  DimsExpr outputDims, xShape;
  createIE->getShapeAsDims(operandAdaptor.getX(), xShape);
  int64_t xRank = xShape.size();
  for (int64_t i = 0; i < xRank; ++i)
    outputDims.emplace_back(QuestionmarkIndexExpr(/*isFloat*/ false));

  auto scalesConstOp = getONNXConstantOp(operandAdaptor.getScales());
  if (scalesConstOp) {
    // Can get the scales as constant.
    auto valueAttr = mlir::dyn_cast<ElementsAttr>(scalesConstOp.getValueAttr());
    if (!valueAttr)
      return op->emitError("Scales constant is not an ElementsAttr");
    for (int64_t i = 0; i < xRank; ++i) {
      if (xShape[i].isLiteral()) {
        // When shape is also constant, replace questionmark by actual value.
        double dim = xShape[i].getLiteral();
        double scale = valueAttr.getValues<FloatAttr>()[i].getValueAsDouble();
        outputDims[i] = LitIE(static_cast<int64_t>(dim * scale));
      }
    }
  }

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXUpsampleOp::verify() {
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getScales()))
    return success();

  auto inputTy = mlir::cast<RankedTensorType>(getX().getType());
  int32_t inputRank = inputTy.getShape().size();

  // Safety checks on scale argument
  auto scalesTy = mlir::cast<RankedTensorType>(getScales().getType());
  if (scalesTy.getShape().size() != 1) {
    return emitError("Scales tensor must be rank 1");
  }
  if (scalesTy.getShape()[0] != inputRank) {
    return emitError("Input tensor rank doesn't match scales tensor shape");
  }

  // Extract the scale values
  auto scalesConstOp = getONNXConstantOp(getScales());
  if (!scalesConstOp) {
    return success();
  }
  auto valueAttr = mlir::dyn_cast<ElementsAttr>(scalesConstOp.getValueAttr());
  if (!valueAttr) {
    return emitError("Scales constant is not an ElementsAttr");
  }

  int scaleIdx = 0;
  for (auto it = valueAttr.getValues<FloatAttr>().begin();
       it != valueAttr.getValues<FloatAttr>().end(); ++it) {
    if (scaleIdx >= inputRank) {
      return emitError("Scales tensor shape doesn't match # of scale values");
    }
    scaleIdx++;
  }
  if (scaleIdx != inputRank) {
    return emitError("Scales tensor shape doesn't match # of scale values");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXUpsampleOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getScales()))
    return success();

  Type elementType =
      mlir::cast<RankedTensorType>(getX().getType()).getElementType();
  ONNXUpsampleOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXUpsampleOp>;
} // namespace onnx_mlir
