/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_MLIR_CONV_OPT_H
#define ONNX_MLIR_CONV_OPT_H

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Exports the ConvOptONNXToONNXPass patterns.
void getConvOptONNXToONNXPatterns(
    bool enableSimdDataLayoutOpt, mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir
#endif
