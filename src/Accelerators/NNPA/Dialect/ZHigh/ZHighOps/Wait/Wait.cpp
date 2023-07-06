/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Wait.cpp - ZHigh Operations------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
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

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//
LogicalResult ZHighWaitOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

} // namespace zhigh
} // namespace onnx_mlir
