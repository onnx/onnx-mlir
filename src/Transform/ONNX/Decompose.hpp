/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

void getDecomposeONNXToONNXPatterns(mlir::RewritePatternSet &patterns);

}
