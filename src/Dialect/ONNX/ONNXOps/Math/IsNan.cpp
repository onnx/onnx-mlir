/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ IsNan.cpp - ONNX Operations -----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect IsNan operation.
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

LogicalResult ONNXIsNaNOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  ONNXIsNaNOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(operandAdaptor.X()))
    return success();

  IntegerType i1Type = IntegerType::get(getContext(), 1, IntegerType::Signless);
  updateType(getResult(), getShape(X().getType()), i1Type);
  return success();
}
