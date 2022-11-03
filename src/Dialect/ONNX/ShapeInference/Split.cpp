/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Split.cpp - Shape Inference for Split Op ---------------===//
//
// This file implements shape inference for the ONNX Split Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

