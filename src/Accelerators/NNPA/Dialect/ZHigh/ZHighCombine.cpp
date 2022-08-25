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
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace {

// Check if all values are produced by ZHighUnstickOp.
bool areProducedByUnstickOp(ValueRange values) {
  return llvm::all_of(values, [](Value v) {
    using namespace onnx_mlir::zhigh;
    // Block argument.
    if (v.isa<BlockArgument>())
      return false;
    // Are produced by ZHighUnstickOp.
    if (isa<ZHighUnstickOp>(v.getDefiningOp())) {
      // Only support LAYOUT_4D and LAYOUT_NHWC at this moment. They have the
      // same stickification scheme.
      Value stickifiedVal = cast<ZHighUnstickOp>(v.getDefiningOp()).In();
      ZTensorEncodingAttr::DataLayout layout =
          getZTensorLayout(stickifiedVal.getType());
      return (layout == ZTensorEncodingAttr::DataLayout::_4D ||
              layout == ZTensorEncodingAttr::DataLayout::NHWC);
    }
    return false;
  });
}

// Check if there are no pads along the given axis when stickifying values by
// using the given layout.
bool haveNoPadsWhenStickified(ValueRange values, StringAttr stickToLayout,
    StringAttr stickFromLayout, IntegerAttr axisAttr) {
  if (!stickToLayout)
    return false;
  // Only support LAYOUT_4D and LAYOUT_NHWC at this moment. They have the same
  // stickification scheme.
  if (!(stickToLayout.getValue().equals_insensitive(onnx_mlir::LAYOUT_NHWC) ||
          stickToLayout.getValue().equals_insensitive(onnx_mlir::LAYOUT_4D)))
    return false;
  // Only support C dimension at this moment.
  int CAxis = 3;
  if (stickFromLayout &&
      stickFromLayout.getValue().equals_insensitive(onnx_mlir::LAYOUT_NCHW))
    CAxis = 1;
  if (axisAttr.getValue().getSExtValue() != CAxis)
    return false;

  // C dimension is tiled by 64 when stickified. Hence, checking `C mod 64` for
  // padding.
  // TODO: get this info from affine_map that is used for stickiyfing NHWC.
  return llvm::all_of(values, [](Value v) {
    if (onnx_mlir::isRankedShapedType(v.getType())) {
      onnx_mlir::zhigh::ZHighUnstickOp unstickOp =
          cast<onnx_mlir::zhigh::ZHighUnstickOp>(v.getDefiningOp());
      StringAttr toLayout = unstickOp.toLayoutAttr();
      ArrayRef<int64_t> dims = onnx_mlir::getShape(v.getType());
      if (toLayout &&
          toLayout.getValue().equals_insensitive(onnx_mlir::LAYOUT_NCHW))
        // NCHW, C is at 1.
        return (dims[1] % 64 == 0);
      else
        // NHWC, C is at 3.
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

IntegerAttr getConcatAxis(PatternRewriter &rewriter, StringAttr toLayout,
    StringAttr fromLayout, IntegerAttr axisAttr) {
  int axis = axisAttr.getValue().getSExtValue();
  SmallVector<int, 4> NCHWtoNHWC = {0, 3, 1, 2};
  if (fromLayout &&
      fromLayout.getValue().equals_insensitive(onnx_mlir::LAYOUT_NCHW))
    return rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, true), NCHWtoNHWC[axis]);
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
