/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_MLIR_CONST_PROP_H
#define ONNX_MLIR_CONST_PROP_H

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Exports the ConstPropONNXToONNXPass patterns.
void getConstPropONNXToONNXPatterns(mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir
#endif