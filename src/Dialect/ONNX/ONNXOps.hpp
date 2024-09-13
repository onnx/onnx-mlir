/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- ONNXOps.hpp - ONNX Operations -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file defines ONNX operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_OPS_H
#define ONNX_MLIR_ONNX_OPS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "src/Dialect/ONNX/ONNXAttributes.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXTraits.hpp"
#include "src/Dialect/ONNX/ONNXTypes.hpp"
#include "src/Interface/HasOnnxSubgraphOpInterface.hpp"
#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"

namespace mlir {
// OpSet level supported by onnx-mlir
static constexpr int CURRENT_ONNX_OPSET = 20;
} // end namespace mlir

#define GET_OP_CLASSES
#include "src/Dialect/ONNX/ONNXOps.hpp.inc"
#endif
