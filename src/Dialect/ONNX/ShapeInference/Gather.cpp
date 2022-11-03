/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ Gather.cpp - Shape Inference for Gather Op --------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX Gather Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

