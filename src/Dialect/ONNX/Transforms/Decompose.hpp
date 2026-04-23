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
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

// Exports the DecomposeONNXToONNXPass patterns. They are all plain rewrite
// patterns that can be used with any PatternRewriter, not conversion patterns.
void getDecomposeONNXToONNXPatterns(
    mlir::RewritePatternSet &patterns, bool enableConvToMatmul = true);

// Check if Conv should be decomposed to Im2Col+MatMul.
// Returns true if the Conv operation meets the criteria for decomposition:
// - Has shape information and static weight shape
// - Is a 2D convolution (rank=4: N×C×H×W)
// - Has group=1 (no grouped convolutions)
// - Is NOT a 1×1 convolution (those are handled elsewhere)
bool shouldDecomposeConvToIm2Col(mlir::ONNXConvOp convOp);

// Add Conv to Im2Col decomposition pattern to the pattern set.
// This pattern transforms Conv into Im2Col+MatMul+Reshape for non-1x1 2D
// convolutions with group=1.
void addConvToIm2ColPattern(mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir
#endif
