//===- onnx_ops.hpp - MLIR ONNX Operations --------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines ONNX operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

#include "src/Pass/ShapeInferenceInterface.hpp"

namespace mlir {

class ONNXOpsDialect : public Dialect {
 public:
  ONNXOpsDialect(MLIRContext* context);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static StringRef getDialectNamespace() { return "onnx"; }
};

/// Include the auto-generated header file containing the declarations of the
/// ONNX operations.
#define GET_OP_CLASSES
#include "src/onnx.hpp.inc"

}  // end namespace mlir

namespace onnx_mlir {}
