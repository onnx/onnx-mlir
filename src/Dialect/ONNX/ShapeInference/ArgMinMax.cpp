/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ArgMinMax.cpp - Shape Inference for ArgMax Op -------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX ArgMin & ArgMax operators.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/Diagnostic.hpp"
#include <type_traits>

using namespace mlir;


// hi alex