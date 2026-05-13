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

// Check if 1x1 Conv should be decomposed to a matmul.
// Applicable for conv 1x1 with group=1, kernel size =1x...x1,
// stride=dilation=1, pad=0.
//
// When hasFastBroadcast1xN is false, then we need to ensure that N=1 (single
// batch only, as multiple back introduce a 1xN broadcast pattern that is
// declared as not supported).
bool shouldDecomposeConv1x1ToMatmul(
    mlir::ONNXConvOp convOp, bool hasFastBroadcast1xN);

// Check if Conv should be decomposed to Im2Col+MatMul.
// Returns true if the Conv operation meets the criteria for decomposition:
// - Has shape information and static weight shape
// - Is a 2D convolution (rank=4: N×C×H×W)
// - Has group=1 (no grouped convolutions)
// - Is NOT a 1×1 convolution (those are handled elsewhere)
//
// When hasFastBroadcast1xN is false, then we need to ensure that N=1 (single
// batch only, as multiple back introduce a 1xN broadcast pattern that is
// declared as not supported).
bool shouldDecomposeConvToIm2Col(
    mlir::ONNXConvOp convOp, bool hasFastBroadcast1xN);

// Add Conv to Im2Col decomposition pattern to the pattern set.
// This pattern transforms Conv into Im2Col+MatMul+Reshape for non-1x1 2D
// convolutions with group=1.
void addConvToMatmulPattern(
    mlir::RewritePatternSet &patterns, bool hasFastBroadcast1xN);

} // namespace onnx_mlir
#endif
