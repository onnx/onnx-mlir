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
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {

// Check if all values are produced by ZHighUnstickOp.
bool areProducedByUnstickOp(ValueRange values, StringAttr layoutAttr) {
  return llvm::all_of(values, [&layoutAttr](Value v) {
    return isa<onnx_mlir::zhigh::ZHighUnstickOp>(v.getDefiningOp());
  });
}

// Check if there are no pads along the given axis when stickifying values by
// using the given layout.
bool haveNoPadsWhenStickified(
    ValueRange values, StringAttr layoutAttr, IntegerAttr axisAttr) {
  if (!layoutAttr)
    return false;
  // Only support LAYOUT_NHWC at this moment.
  if (!layoutAttr.getValue().equals_insensitive(onnx_mlir::LAYOUT_NHWC))
    return false;
  // Only support C dimension at this moment.
  if (axisAttr.getValue().getSExtValue() != 3)
    return false;
  // C dimension is tiled by 64 when stickified. Hence, checking `C mod 64` for
  // padding.
  // TODO: get this info from affine_map that is used for stickiyfing NHWC.
  return llvm::all_of(values, [](Value v) {
    if (v.getType().isa<ShapedType>() &&
        v.getType().cast<ShapedType>().hasRank()) {
      ArrayRef<int64_t> dims = v.getType().cast<ShapedType>().getShape();
      return (dims[3] % 64 == 0);
    }
    return false;
  });
}

SmallVector<Value, 4> getStickifiedInputs(
    PatternRewriter &rewriter, Location loc, ValueRange values) {
  SmallVector<Value, 4> stickfiedValues;
  for (Value v : values)
    stickfiedValues.emplace_back(v.getDefiningOp()->getOperands()[0]);
  return stickfiedValues;
}

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Dialect/ZHigh/ONNXZHighCombine.inc"
} // end anonymous namespace

namespace onnx_mlir {
namespace zhigh {

/// Register optimization patterns as "canonicalization" patterns
/// on the ZHighStickOp.
void ZHighStickOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<StickUnstickRemovalPattern>(context);
  results.insert<NoneTypeStickRemovalPattern>(context);
  results.insert<ReplaceONNXConcatByZHighConcatPattern>(context);
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
