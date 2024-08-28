/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Softmax.cpp - ZHigh Operations --------------------===//
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

LogicalResult ZHighSoftmaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

} // namespace zhigh
} // namespace onnx_mlir
