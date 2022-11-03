/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- OneHot.cpp - Shape Inference for OneHot Op -------------===//
//
// This file implements shape inference for the ONNX OneHot Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

