/*
 * SPDX-License-Identifier: Apache-2.0
 */

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

/// Test if two axis arrays contain the same values or not.
bool AreTheSameAxisArray(int64_t rank, ArrayAttr lhsAttr, ArrayAttr rhsAttr) {
  // false if one of the array attributes is null.
  if (!(lhsAttr) || !(rhsAttr))
    return false;

  SmallVector<int64_t, 4> lhs;
  for (auto attr : lhsAttr.getValue()) {
    int64_t axis = attr.cast<IntegerAttr>().getInt();
    if (axis < 0)
      axis += rank;
    lhs.emplace_back(axis);
  }

  int64_t rhsSize = 0;
  for (auto attr : rhsAttr.getValue()) {
    int64_t axis = attr.cast<IntegerAttr>().getInt();
    if (axis < 0)
      axis += rank;
    // false if axis is not in the lhs. Early stop.
    if (!llvm::any_of(lhs, [&](int64_t lhsAxis) { return lhsAxis == axis; }))
      return false;
    rhsSize++;
  }

  // false if having different number of elements.
  if (lhs.size() != rhsSize)
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

/// on the ONNXDropoutOp.
void ONNXDropoutOp::getCanonicalizationPatterns(
    OwningRewritePatternList &result, MLIRContext *context) {
  result.insert<DropoutEliminationPattern>(context);
}

/// on the ONNXSqueezeOp.
void ONNXSqueezeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &result, MLIRContext *context) {
  result.insert<RemoveSqueezeUnsqueezePattern>(context);
}

/// on the ONNXUnsqueezeOp.
void ONNXUnsqueezeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &result, MLIRContext *context) {
  result.insert<RemoveUnsqueezeSqueezePattern>(context);
}
