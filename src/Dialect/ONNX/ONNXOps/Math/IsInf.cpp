/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ IsInf.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
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
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXIsInfOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  IntegerType i1Type = IntegerType::get(getContext(), 1, IntegerType::Signless);
  return inferShapeForUnaryOps(getOperation(), i1Type);
}
