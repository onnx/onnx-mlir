/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Unsqueeze.cpp - Shape Inference for Unsqueeze Op ----------===//
//
// This file implements shape inference for the ONNX Unsqueeze Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

