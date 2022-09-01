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
using namespace onnx_mlir;

namespace {

// Check if all values are produced by ZHighUnstickOp.
bool areProducedByUnstickOp(
    PatternRewriter &rewriter, ValueRange values, StringAttr layout) {
  // Only support LAYOUT_4D and LAYOUT_NHWC at this moment. They have the same
  // stickification scheme.
  if (!(isNHWCLayout(layout) || is4DLayout(layout)))
    return false;

  return llvm::all_of(values, [&](Value v) {
    using namespace onnx_mlir::zhigh;
    // Block argument.
    if (v.isa<BlockArgument>())
      return false;
    // Not produced by ZHighUnstickOp.
    if (!isa<ZHighUnstickOp>(v.getDefiningOp()))
      return false;

    // Only support LAYOUT_4D and LAYOUT_NHWC at this moment. They have the
    // same stickification scheme.
    Value stickifiedVal = cast<ZHighUnstickOp>(v.getDefiningOp()).In();
    StringAttr valueLayout = convertDataLayoutToStringAttr(
        rewriter, getZTensorLayout(stickifiedVal.getType()));
    return (valueLayout == layout);
  });
}

// Check if there are no pads along the given axis when stickifying values by
// using the given layout.
bool haveNoPadsWhenStickified(
    ValueRange values, StringAttr layoutAttr, IntegerAttr axisAttr) {
  if (!layoutAttr)
    return false;
  // Only support LAYOUT_4D and LAYOUT_NHWC at this moment. They have the same
  // stickification scheme.
  if (!(isNHWCLayout(layoutAttr) || is4DLayout(layoutAttr)))
    return false;
  // Only support C dimension at this moment.
  int CAxis = 3; // C is at 3 for 4D and NHWC.
  if (isNHWCLayout(layoutAttr))
    // Value is NCHW that will be directly stickified to NHWC. So C is at 1.
    CAxis = 1;
  if (axisAttr.getValue().getSExtValue() != CAxis)
    return false;

  // C dimension is tiled by 64 when stickified. Hence, checking `C mod 64` for
  // padding.
  // TODO: get this info from affine_map that is used for stickiyfing NHWC.
  return llvm::all_of(values, [&layoutAttr](Value v) {
    if (v.getType().isa<ShapedType>() &&
        v.getType().cast<ShapedType>().hasRank()) {
      ArrayRef<int64_t> dims = v.getType().cast<ShapedType>().getShape();
      if (isNHWCLayout(layoutAttr))
        // Value is NCHW that will be directly unstickified from NHWC.
        // NCHW, C is at 1.
        return (dims[1] % 64 == 0);
      else
        // 4D (similar to NHWC), C is at 3.
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

IntegerAttr getStickifiedConcatAxis(
    PatternRewriter &rewriter, StringAttr layout, IntegerAttr axisAttr) {
  int axis = axisAttr.getValue().getSExtValue();
  if (isNHWCLayout(layout)) {
    SmallVector<int, 4> NCHWtoNHWC = {0, 3, 1, 2};
    return rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, true), NCHWtoNHWC[axis]);
  }
  return axisAttr;
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
  results.insert<ReplaceONNXLeakyReluPattern>(context);
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
