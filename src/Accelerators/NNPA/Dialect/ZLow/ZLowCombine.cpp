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

// A helper function to remove zlow operations whose outputs are ones in the
// input operands. This function checks if the outputs have no other use rather
// than the current zlow operation then it removes the zlow operation.
static LogicalResult removeUnusedOp(
    PatternRewriter &rewriter, Operation *op, ArrayRef<int64_t> resultIndices) {
  SmallVector<Value> results;
  for (int64_t i : resultIndices)
    results.emplace_back(op->getOperands()[i]);
  // Check if this operation is the only one that uses the output buffers.
  bool allHasOneUse =
      llvm::all_of(results, [](Value v) { return v.hasOneUse(); });
  if (allHasOneUse) {
    rewriter.eraseOp(op);
    return success();
  } else {
    return failure();
  }
}

class RemoveUnusedStickOpPattern : public OpRewritePattern<ZLowStickOp> {
public:
  using OpRewritePattern<ZLowStickOp>::OpRewritePattern;

  RemoveUnusedStickOpPattern(MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(
      ZLowStickOp stickOp, PatternRewriter &rewriter) const override {
    return removeUnusedOp(rewriter, stickOp.getOperation(), {1});
  }
};

class RemoveUnusedUnstickOpPattern : public OpRewritePattern<ZLowUnstickOp> {
public:
  using OpRewritePattern<ZLowUnstickOp>::OpRewritePattern;

  RemoveUnusedUnstickOpPattern(MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(
      ZLowUnstickOp unstickOp, PatternRewriter &rewriter) const override {
    return removeUnusedOp(rewriter, unstickOp.getOperation(), {1});
  }
};

/// Register optimization patterns as "canonicalization" patterns on the
/// ZLowDummyOp.
void ZLowDummyOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveDummyOpPattern>(context);
}

/// ZLowStickOp
void ZLowStickOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveUnusedStickOpPattern>(context);
}

/// ZLowStickOp
void ZLowUnstickOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveUnusedUnstickOpPattern>(context);
}

/// ZLowConvertF32ToDLF16Op.
void ZLowConvertF32ToDLF16Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<DLF16ConversionOpPattern>(context);
}

} // namespace zlow
} // namespace onnx_mlir
