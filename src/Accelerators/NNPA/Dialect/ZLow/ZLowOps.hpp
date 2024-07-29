/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ZLowOps.hpp - ZLow Operations ---------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file defines the ZLow operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ZLOW_H
#define ONNX_MLIR_ZLOW_H

#include <map>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

/// Include the auto-generated header files containing the declarations of the
/// ZLow dialect and operations.
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowDialect.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp.inc"
#endif
