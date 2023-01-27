/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ IsInf.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect IsInf operation.
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
LogicalResult ONNXIsInfOpShapeHelper::computeShape() {
  ONNXIsInfOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  // Get info about "X" data operand.
  Value X = operandAdaptor.X();

  // detect_negative is an optional attribute and should have default value
  // of 1.
  int64_t detect_negative = operandAdaptor.detect_negative();
  bool isDetectNegative = (detect_negative == 1);

  // detect_positive is an optional attribute and should have default value
  // of 1.
  int64_t detect_positive = operandAdaptor.detect_positive();
  bool isDetectPositive = (detect_positive == 1);

  // If negative infinity should be mapped to false, set the attribute to 0
  if (X < 0) {
     !isDetectNegative
  }

  // If positive infinity should be mapped to false, set the attribute to 0
  if (X > 0) {
    !isDetectPositive
  }


  return success();
}
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXIsInfOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  IntegerType i1Type = IntegerType::get(getContext(), 1,
  IntegerType::Signless); return inferShapeForUnaryOps(getOperation(),
  i1Type);
}
