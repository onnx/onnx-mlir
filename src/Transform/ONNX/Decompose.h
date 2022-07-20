/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXDecompose.h - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to decompose an ONNX operation into
// composition of other ONNX operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace onnx_mlir {

void populateDecomposingONNXBeforeMhloPatterns(
    RewritePatternSet &, MLIRContext *);

}