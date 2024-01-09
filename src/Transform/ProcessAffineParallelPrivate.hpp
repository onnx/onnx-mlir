/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===- ProcessAffineParallelPrivate.hpp - Handle parallel private data ----===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file adds alloca scope to affine parallel for loop for proper handling of
// parallel for private data structures.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Exports the RecomposeONNXToONNXPass patterns. They are all plain rewrite
// patterns that can be used with any PatternRewriter, not conversion patterns.
void getParallelPrivateAffineToAffinePatterns(
    mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir
