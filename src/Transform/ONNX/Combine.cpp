//===--------- ONNXCombine.cpp - ONNX High Level Optimizer ----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ONNX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <numeric>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Support for transpose patterns.
//===----------------------------------------------------------------------===//

/// Compute the combined permute pattern from a pair of permute patterns.
ArrayAttr CombinedTransposePattern(PatternRewriter &rewriter,
    ArrayAttr firstPermAttr, ArrayAttr secondPermAttr) {
  // Read first permute vectors.
  SmallVector<int64_t, 4> initialPerm;
  for (auto firstPermVal : firstPermAttr.getValue())
    initialPerm.emplace_back(firstPermVal.cast<IntegerAttr>().getInt());
  // Read second permute vector. Use it as an index in the first permute
  // vector.
  SmallVector<int64_t, 4> resPerm;
  for (auto secondPermVal : secondPermAttr.getValue()) {
    auto index = secondPermVal.cast<IntegerAttr>().getInt();
    resPerm.emplace_back(initialPerm[index]);
  }
  // Convert to Array of Attributes.
  ArrayRef<int64_t> resPermRefs(resPerm);
  return rewriter.getI64ArrayAttr(resPermRefs);
}

/// Test if the permute pattern correspond to an identity pattern.
/// Identity patterns are {0, 1, 2, ... , rank -1}.
bool IsIdentityPermuteVector(ArrayAttr permAttr) {
  int64_t currentIndex = 0;
  for (auto permVal : permAttr.getValue())
    if (permVal.cast<IntegerAttr>().getInt() != currentIndex++)
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Pattern definition.
//===----------------------------------------------------------------------===//

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Transform/ONNX/ONNXCombine.inc"
} // end anonymous namespace

/// Register optimization patterns as "canonicalization" patterns
/// on the ONNXMatMultOp.
void ONNXAddOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<MulAddToGemmOptPattern>(context);
}

void ONNXGemmOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FuseGemmFollowedByAddition>(context);
}
/// on the ONNXIdentityOp.
void ONNXIdentityOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<IdentityEliminationPattern>(context);
}

/// on the ONNXPadConstantValueOp.
void ONNXPadConstantValueOp::getCanonicalizationPatterns(
    OwningRewritePatternList &result, MLIRContext *context) {
  result.insert<ConstantPadPattern>(context);
}

/// on the ONNXCastOp.
void ONNXCastOp::getCanonicalizationPatterns(
    OwningRewritePatternList &result, MLIRContext *context) {
  result.insert<CastEliminationPattern>(context);
}

/// on the ONNXTransposeOp.
void ONNXTransposeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &result, MLIRContext *context) {
  result.insert<FuseTransposePattern>(context);
  result.insert<RemoveIdentityTransposePattern>(context);
}
