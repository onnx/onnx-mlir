/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- ONNXOps.hpp - ONNX Operations -------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines ONNX operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <string>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXTypes.hpp"
#include "src/Interface/HasOnnxSubgraphOpInterface.hpp"
#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"

namespace mlir {

// OpSet level supported by onnx-mlir
static constexpr int CURRENT_ONNX_OPSET = 13;
} // end namespace mlir

/// Include the auto-generated header file containing the declarations of the
/// ONNX operations.
#define GET_OP_CLASSES
#include "src/Dialect/ONNX/ONNXOps.hpp.inc"
