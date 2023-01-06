/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Upsample.cpp - ONNX Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Upsample operation
// (deprecated).
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXUpsampleOp::verify() {
  if (!X().getType().isa<RankedTensorType>()) {
    return success();
  }
  if (!scales().getType().isa<RankedTensorType>()) {
    return success();
  }

  auto inputTy = X().getType().cast<RankedTensorType>();
  int32_t inputRank = inputTy.getShape().size();

  // Safety checks on scale argument
  auto scalesTy = scales().getType().cast<RankedTensorType>();
  if (scalesTy.getShape().size() != 1) {
    return emitError("Scales tensor must be rank-1");
  }
  if (scalesTy.getShape()[0] != inputRank) {
    return emitError("Input tensor rank doesn't match scales tensor shape");
  }

  // Extract the scale values
  auto scalesConstOp = getONNXConstantOp(scales());
  if (!scalesConstOp) {
    return success();
  }
  auto valueAttr = scalesConstOp.valueAttr().dyn_cast<DenseElementsAttr>();
  if (!valueAttr) {
    return emitError("Scales constant is not a DenseElementsAttr");
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
  if (!X().getType().isa<RankedTensorType>()) {
    return success();
  }
  if (!scales().getType().isa<RankedTensorType>()) {
    return success();
  }

  auto inputTy = X().getType().cast<RankedTensorType>();
  int32_t inputRank = inputTy.getShape().size();

  SmallVector<int64_t, 4> outputDims(inputRank, -1);

  // Extract the scale values
  auto scalesConstOp = getONNXConstantOp(scales());
  if (!scalesConstOp) {
    return success();
  }
  auto valueAttr = scalesConstOp.valueAttr().dyn_cast<DenseElementsAttr>();
  if (!valueAttr) {
    return emitError("Scales constant is not a DenseElementsAttr");
  }
  int scaleIdx = 0;
  // Why are the scale values float's?
  for (auto it = valueAttr.getValues<FloatAttr>().begin();
       it != valueAttr.getValues<FloatAttr>().end(); ++it) {
    outputDims[scaleIdx++] = (int)((*it).getValueAsDouble());
  }

  // Compute and set the output shape
  for (int i = 0; i < inputRank; ++i) {
    outputDims[i] *= inputTy.getShape()[i];
  }
  getResult().setType(
      RankedTensorType::get(outputDims, inputTy.getElementType()));

  return success();
}
