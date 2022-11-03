/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Compress.cpp - Shape Inference for Compress Op -----------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX Compress Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/Diagnostic.hpp"

using namespace mlir;

// hi alex