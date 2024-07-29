/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- ShapeInferenceOpInterface.hpp - Definition for ShapeInference ---===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_SHAPE_INFERENCE_INTERFACE_H
#define ONNX_MLIR_SHAPE_INFERENCE_INTERFACE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

/// Include the auto-generated declarations.
#include "src/Interface/ShapeInferenceOpInterface.hpp.inc"
#endif
