/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ONNXTypes.hpp - ONNX Types ---------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file defines types in ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_TYPES_H
#define ONNX_MLIR_ONNX_TYPES_H

#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "src/Dialect/ONNX/ONNXTypes.hpp.inc"
#endif
