/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ZHighCombine.cpp - ZHigh High Level Optimizer ----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ZHigh dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Dialect/ZHigh/ZHighCombine.inc"
} // end anonymous namespace

namespace onnx_mlir {
namespace zhigh {

/// Register optimization patterns as "canonicalization" patterns
/// on the ZHighStickOp.
void ZHighStickOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<StickUnstickRemovalPattern>(context);
  results.insert<NoneTypeStickRemovalPattern>(context);
}

/// Register optimization patterns as "canonicalization" patterns
/// on the ZHighUnstickOp.
void ZHighUnstickOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<UnstickStickRemovalPattern>(context);
  results.insert<SigmoidLayoutChangingPattern>(context);
}

} // namespace zhigh
} // namespace onnx_mlir
