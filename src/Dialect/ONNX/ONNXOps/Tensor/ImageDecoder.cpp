/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- ImageDecoder.cpp - ONNX Operations -----------------===//
//
// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
// =============================================================================
//
// This file provides definition of ONNX dialect ImageDecoder operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace OpTrait::util;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXImageDecoderOpShapeHelper::computeShape() {
  // All 3 dims (H, W, C) depend on the encoded image content, so are unknown at
  // compile time.
  DimsExpr outputDims(3, QuestionmarkIndexExpr(/*isFloat=*/false));
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

LogicalResult ONNXImageDecoderOp::inferShapes(
    std::function<void(Region &)> /*doShapeInference*/) {
  ONNXImageDecoderOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(
      getElementTypeOrSelf(getEncodedStream().getType()));
}

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXImageDecoderOp>;
} // namespace onnx_mlir
