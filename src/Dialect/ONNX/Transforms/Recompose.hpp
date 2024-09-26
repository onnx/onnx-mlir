/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXRecompose.hpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to recompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the recomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_RECOMPOSE_H
#define ONNX_MLIR_RECOMPOSE_H

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Exports the RecomposeONNXToONNXPass patterns. They are all plain rewrite
// patterns that can be used with any PatternRewriter, not conversion patterns.
void getRecomposeONNXToONNXPatterns(mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir
#endif
