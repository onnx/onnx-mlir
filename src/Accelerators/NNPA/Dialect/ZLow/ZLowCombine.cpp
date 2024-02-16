/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- ZLowCombine.cpp - ZLow High Level Optimizer ----------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ZLow dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Dialect/ZLow/ONNXZLowCombine.inc"
} // end anonymous namespace

namespace onnx_mlir {
namespace zlow {

/// Register optimization patterns as "canonicalization" patterns on the
/// ZLowDummyOp.
void ZLowDummyOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveDummyOpPattern>(context);
}

/// ZLowConvertF32ToDLF16Op.
void ZLowConvertF32ToDLF16Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<DLF16ConversionOpPattern>(context);
}

} // namespace zlow
} // namespace onnx_mlir
