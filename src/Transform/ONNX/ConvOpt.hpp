/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Exports the ConvOptONNXToONNXPass patterns.
void getConvOptONNXToONNXPatterns(
    bool enableSimdDataLayoutOpt, mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir
