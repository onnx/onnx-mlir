/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ MenReduce2D.cpp - ZHigh Operations ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace zhigh {

LogicalResult ZHighMeanReduce2DOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()))
    return success();

  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = inputType.getShape();

  // Input is NHWC, and H and W are reduction dimensions.
  updateType(getResult(), {shape[0], 1, 1, shape[3]},
      inputType.getElementType(), inputType.getEncoding());
  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
