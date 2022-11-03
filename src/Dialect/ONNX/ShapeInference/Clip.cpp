/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Clip.cpp - Shape Inference for Clip Op ---------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX Clip operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

