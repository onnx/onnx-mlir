//===----------------- MLONNXOps.hpp - ONNX ML Operations ----_------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines ONNX ML operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <string>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

#include "src/Interface/PromotableConstOperandsOpInterface.hpp"
#include "src/Interface/ShapeInferenceInterface.hpp"

namespace mlir {

class MLONNXOpsDialect : public Dialect {
public:
  MLONNXOpsDialect(MLIRContext *context);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static StringRef getDialectNamespace() { return "onnx"; }
};

/// Include the auto-generated header file containing the declarations of the
/// ONNX operations.
#define GET_OP_CLASSES
#include "src/Dialect/MLONNX/MLONNXOps.hpp.inc"

} // end namespace mlir

namespace onnx_mlir {}
