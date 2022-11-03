/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ Reshape.cpp - Shape Inference for Reshape Op ------------===//
//
// This file implements shape inference for the ONNX Reshape Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

