/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Exports the DecomposeONNXToONNXPass patterns. They are all plain rewrite
// patterns that can be used with any PatternRewriter, not conversion patterns.
void getDecomposeONNXToONNXPatterns(mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir
