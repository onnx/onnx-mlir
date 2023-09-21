/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToZHigh.hpp - ONNX dialect to ZHigh lowering -------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ONNX operations to a combination of
// ONNX and ZHigh operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

// Exports ONNXtoZHigh patterns.
void getONNXToZHighOneOpPatterns(mlir::RewritePatternSet &patterns);
void getONNXToZHighMultipleOpPatterns(mlir::RewritePatternSet &patterns);

// Exports ONNXtoZHigh dynamically legal checks.
void getONNXToZHighOneOpDynamicallyLegal(
    mlir::ConversionTarget *target, const DimAnalysis *dimAnalysis);

} // namespace onnx_mlir
