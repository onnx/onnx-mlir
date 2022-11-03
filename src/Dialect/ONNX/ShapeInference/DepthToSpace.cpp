/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ DepthToSpace.cpp - Shape Inference for DepthToSpace Op --------===//
//
// This file implements shape inference for the ONNX DepthToSpace Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

// hi alex