/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Flatten.cpp - Shape Inference for Flatten Op --------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX Flatten Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

