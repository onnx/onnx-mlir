/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Transpose.cpp - Shape Inference for Transpose Op ----------===//
//
// This file implements shape inference for the ONNX Transpose Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

