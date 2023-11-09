/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- RewriteONNXForZHigh.hpp - Rewrite ONNX ops for ZHigh lowering ----===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file implements pass for rewriting of ONNX operations to generate
// combination of ONNX and ZHigh operations.

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

// Exports RewriteONNXForZHigh patterns.
void getRewriteONNXForZHighPatterns(mlir::RewritePatternSet &patterns,
    DimAnalysis *dimAnalysis, std::string nnpaParallelOpt = "");

// Exports RewriteONNXForZHigh dynamically legal checks.
void getRewriteONNXForZHighDynamicallyLegal(mlir::ConversionTarget *target,
    const DimAnalysis *dimAnalysis, std::string nnpaParallelOpt = "");

} // namespace onnx_mlir
