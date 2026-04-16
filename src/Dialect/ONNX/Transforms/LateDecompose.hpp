/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- LateDecompose.hpp - Late Decomposition Patterns -----------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declarations for the LateDecompose pass.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_LATE_DECOMPOSE_H
#define ONNX_MLIR_LATE_DECOMPOSE_H

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Populate patterns for late decomposition.
void getLateDecomposePatterns(mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir

#endif // ONNX_MLIR_LATE_DECOMPOSE_H

// Made with Bob
