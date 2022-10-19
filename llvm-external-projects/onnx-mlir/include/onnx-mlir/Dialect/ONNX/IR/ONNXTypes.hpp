/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ONNXTypes.hpp - ONNX Types ---------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines types in ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "onnx-mlir/Dialect/ONNX/IR/ONNXDialect.hpp"

#define GET_TYPEDEF_CLASSES
#include "onnx-mlir/Dialect/ONNX/IR/ONNXTypes.hpp.inc"
