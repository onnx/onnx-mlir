/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ NonZero.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect NonZero operation.
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

LogicalResult ONNXNonZeroOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto builder = Builder(getContext());
  Type inputType = getOperand().getType();
  if (!inputType.isa<RankedTensorType>())
    return success();
  SmallVector<int64_t, 2> dims;
  // The first dimension size is the rank of the input.
  dims.emplace_back(inputType.cast<RankedTensorType>().getRank());
  // The second dimension size is the number of nonzero values in the input.
  // So this dimension size is always unknown at compile time.
  dims.emplace_back(-1);
  getResult().setType(RankedTensorType::get(dims, builder.getI64Type()));
  return success();
}
