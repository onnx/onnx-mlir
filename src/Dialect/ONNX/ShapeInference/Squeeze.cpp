/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ Squeeze.cpp - Shape Inference for Squeeze Op ------------===//
//
// This file implements shape inference for the ONNX Squeeze Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

