/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ HardMax.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect HardMax operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXHardmaxOp::verify() {
  ONNXHardmaxOpAdaptor operandAdaptor(*this);
  Value input = operandAdaptor.input();
  if (!hasShapeAndRank(input))
    return success(); // Won't be able to do any checking at this stage.

  // axis attribute must be in the range [-r,r-1], where r = rank(input).
  int64_t axisValue = axis();
  int64_t inputRank = input.getType().cast<ShapedType>().getRank();
  if (axisValue < -inputRank || axisValue >= inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisValue,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXHardmaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(input()))
    return success();

  auto inputType = input().getType().cast<ShapedType>();
  int64_t inputRank = inputType.getRank();
  int64_t axisValue = axis();

  // axis attribute must be in the range [-r,r], where r = rank(input).
  if (axisValue < -inputRank || axisValue > inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisValue,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));

  return inferShapeForUnaryOps(this->getOperation());
}
