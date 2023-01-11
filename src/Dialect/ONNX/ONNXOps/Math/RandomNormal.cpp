/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ RandomNormal.cpp - ONNX Operations ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect RandomNormal operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXRandomNormalOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto outputShape = shape();
  auto elementTypeID = dtype();

  SmallVector<int64_t, 4> outputDims;
  auto spatialRank = ArrayAttrSize(outputShape);
  for (unsigned long i = 0; i < spatialRank; ++i) {
    int64_t dimension = ArrayAttrIntVal(outputShape, i);
    if (dimension < 0)
      return emitError("Random normal tensor has dynamic dimension.");
    outputDims.emplace_back(dimension);
  }

  RankedTensorType outputTensorType =
      RankedTensorType::get(outputDims, FloatType::getF32(getContext()));
  if (elementTypeID == 0)
    outputTensorType =
        RankedTensorType::get(outputDims, FloatType::getF16(getContext()));
  else if (elementTypeID == 2)
    outputTensorType =
        RankedTensorType::get(outputDims, FloatType::getF64(getContext()));

  getResult().setType(outputTensorType);
  return success();
}
