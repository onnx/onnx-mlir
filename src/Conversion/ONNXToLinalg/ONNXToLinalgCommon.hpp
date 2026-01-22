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
inline bool shouldConvertToLinalg(
    mlir::Operation *op, const std::string &linalgOps, bool useLinalgPath) {
  // When convert-onnx-to-linalg pass is explicitly run (e.g., via
  // onnx-mlir-opt), we default to converting all operations unless linalgOps
  // is explicitly set
  if (linalgOps.empty()) {
    // If linalgOps is not specified, check useLinalgPath flag
    // If useLinalgPath is true, convert all operations to Linalg
    // Otherwise, default to true for onnx-mlir-opt usage (when pass is
    // explicitly run) Note: In onnx-mlir-opt, useLinalgPath may not be
    // initialized, so we default to true
    return true;
  }

  // Get operation name with dialect prefix (e.g., "onnx.MatMul") for future
  // support of other dialects (e.g., "tosa.MatMul", "stablehlo.MatMul")
  // Also support matching without dialect prefix (e.g., "MatMul") for
  // backward compatibility
  std::string opNameWithDialect = op->getName().getStringRef().str();
  std::string opNameWithoutDialect = op->getName().stripDialect().str();

  // Use EnableByRegexOption to check if operation matches
  // Use static variable with lazy initialization to ensure linalgOps is set
  // emptyIsNone=false means empty string enables all (but we already checked)
  static std::string cachedLinalgOps;
  static std::unique_ptr<EnableByRegexOption> linalgOpsMatcher;

  // Reinitialize if linalgOps has changed (shouldn't happen in practice, but
  // safe)
  if (cachedLinalgOps != linalgOps) {
    cachedLinalgOps = linalgOps;
    linalgOpsMatcher = std::make_unique<EnableByRegexOption>(false, linalgOps);
  }

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
