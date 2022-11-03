/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- GatherND.cpp - Shape Inference for GatherND Op ------------===//
//
// This file implements shape inference for the ONNX GatherND Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include <algorithm>

using namespace mlir;

