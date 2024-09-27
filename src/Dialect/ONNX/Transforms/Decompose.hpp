/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXDecompose.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to decompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the decomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DECOMPOSE_H
#define ONNX_MLIR_DECOMPOSE_H

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Exports the DecomposeONNXToONNXPass patterns. They are all plain rewrite
// patterns that can be used with any PatternRewriter, not conversion patterns.
void getDecomposeONNXToONNXPatterns(mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir
#endif
