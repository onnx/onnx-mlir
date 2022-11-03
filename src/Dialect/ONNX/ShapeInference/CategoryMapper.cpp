/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- CategoryMapper.cpp - Shape Inference for CategoryMapper Op -----===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX CategoryMapper operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

