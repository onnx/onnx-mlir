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
// based on the --linalg-ops option. Returns true if the operation name
// matches the specified patterns, or if --linalg-ops is not set and
// --use-linalg-path is enabled.
inline bool shouldConvertToLinalg(mlir::Operation *op) {
  // If --linalg-ops is not specified, fall back to --use-linalg-path behavior
  extern std::string linalgOps;
  extern bool useLinalgPath;

  if (linalgOps.empty()) {
    return useLinalgPath;
  }

  // Get operation name without dialect prefix (e.g., "MatMul" from
  // "onnx.MatMul")
  std::string opName = op->getName().stripDialect().str();

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

  return linalgOpsMatcher->isEnabled(opName);
}

// Math operations
void populateLoweringONNXMatMulOpToLinalgPattern(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *ctx);

} // namespace onnx_mlir
