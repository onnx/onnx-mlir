/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- GatherElements.cpp - Shape Inference for GatherElements Op -----===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX GatherElements Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

