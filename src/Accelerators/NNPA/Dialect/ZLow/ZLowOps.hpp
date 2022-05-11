/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ZLowOps.hpp - ZLow Operations ---------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file defines the ZLow operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace onnx_mlir {
namespace zlow {

class ZLowDialect : public mlir::Dialect {
public:
  ZLowDialect(mlir::MLIRContext *context);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static mlir::StringRef getDialectNamespace() { return "zlow"; }
};

} // namespace zlow
} // namespace onnx_mlir

/// Include the auto-generated header file containing the declarations of the
/// ONNX operations.
#define GET_OP_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp.inc"
