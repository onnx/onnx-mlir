/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ONNXTypes.hpp - ONNX Types ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines types in ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "src/Dialect/ONNX/ONNXTypes.hpp.inc"
