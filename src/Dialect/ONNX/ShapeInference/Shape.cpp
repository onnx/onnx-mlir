/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Shape.cpp - Shape Inference for Shape Op --------------===//
//
// This file implements shape inference for the ONNX Shape Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include <tuple>
#include <utility>

using namespace mlir;

