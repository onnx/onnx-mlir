//===- ONNXToLinalgCommon.hpp - Common functions for ONNX to Linalg ------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code for lowering ONNX operations to Linalg.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <string>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Compiler/OptionUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

// Utility function to check if an operation should be converted to Linalg
// based on the linalg-ops option. Returns true if the operation name
// matches the specified patterns, or if linalgOps is empty and
// useLinalgPath is enabled.
// Note: When convert-onnx-to-linalg pass is explicitly run (e.g., via
// onnx-mlir-opt), we default to converting operations if no options are set.
// The linalgOpsMatcher parameter should be a pointer to an EnableByRegexOption
// instance that is thread-safe (each pattern instance should have its own).
// Note: linalgOpsMatcher is non-const because isEnabled() modifies internal cache.
inline bool shouldConvertToLinalg(mlir::Operation *op,
    EnableByRegexOption *linalgOpsMatcher, bool useLinalgPath) {
  // When linalgOpsMatcher is null or empty, default to converting all operations
  if (!linalgOpsMatcher) {
    return true;
  }

  // Get operation name with dialect prefix (e.g., "onnx.MatMul") for future
  // support of other dialects (e.g., "tosa.MatMul", "stablehlo.MatMul")
  // Also support matching without dialect prefix (e.g., "MatMul") for
  // backward compatibility
  std::string opNameWithDialect = op->getName().getStringRef().str();
  std::string opNameWithoutDialect = op->getName().stripDialect().str();

  // Check both with and without dialect prefix to support:
  // - "onnx.MatMul" or "MatMul" patterns
  // - Future dialects like "tosa.MatMul", "stablehlo.MatMul"
  return linalgOpsMatcher->isEnabled(opNameWithDialect) ||
         linalgOpsMatcher->isEnabled(opNameWithoutDialect);
}

// Math operations
void populateLoweringONNXMatMulOpToLinalgPattern(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *ctx, const std::string &linalgOps, bool useLinalgPath);

} // namespace onnx_mlir
