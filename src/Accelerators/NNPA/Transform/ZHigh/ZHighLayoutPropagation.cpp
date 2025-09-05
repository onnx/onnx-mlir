/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZHighLayoutPropagation.cpp - ZHigh High Level Optimizer ---===//
//
// Copyright 2019-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ZHigh dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace onnx_mlir;
using namespace onnx_mlir::zhigh;

namespace onnx_mlir {
namespace zhigh {

namespace {

//===----------------------------------------------------------------------===//
// Helper functions for this pass
//===----------------------------------------------------------------------===//

/// Check if all values are produced by ZHighUnstickOp with the same layout.
std::pair<bool, StringAttr> areProducedByUnstickOpSameLayout(
    PatternRewriter &rewriter, ValueRange values) {
  // Check the first value and get its layout.
  Value first = values[0];
  if (mlir::isa<BlockArgument>(first) ||
      !isa<ZHighUnstickOp>(first.getDefiningOp()))
    return std::make_pair(false, nullptr);
  Value firstStickifiedVal =
      mlir::cast<ZHighUnstickOp>(first.getDefiningOp()).getIn();
  StringAttr firstLayout = convertZTensorDataLayoutToStringAttr(
      rewriter, getZTensorLayout(firstStickifiedVal.getType()));

  // Check all values.
  bool allTheSame = llvm::all_of(values, [&](Value v) {
    using namespace onnx_mlir::zhigh;
    if (mlir::isa<BlockArgument>(v) || !isa<ZHighUnstickOp>(v.getDefiningOp()))
      return false;
    Value stickifiedVal = mlir::cast<ZHighUnstickOp>(v.getDefiningOp()).getIn();
    StringAttr nextLayout = convertZTensorDataLayoutToStringAttr(
        rewriter, getZTensorLayout(stickifiedVal.getType()));
    return (nextLayout == firstLayout);
  });

  if (allTheSame)
    return std::make_pair(true, firstLayout);
  return std::make_pair(false, nullptr);
}

/// Return zTensors that are unstickified into the given tensors.
SmallVector<Value, 4> getZTensors(
    PatternRewriter &rewriter, Location loc, ValueRange tensors) {
  SmallVector<Value, 4> zTensors;
  for (Value v : tensors)
    zTensors.emplace_back(v.getDefiningOp()->getOperands()[0]);
  return zTensors;
}

/// Return a zTensorType for the given tensor and layout.
Type getZTensorType(
    PatternRewriter &rewriter, Location loc, Value tensor, StringAttr layout) {
  // Borrow ZHighStickOp to infer a zTensor type.
  ZHighStickOp stickOp =
      rewriter.create<ZHighStickOp>(loc, tensor, layout, IntegerAttr());
  (void)stickOp.inferShapes([](Region &region) {});

  Type returnType = stickOp.getOut().getType();
  rewriter.eraseOp(stickOp);

  return returnType;
}

//===----------------------------------------------------------------------===//
// ZHigh layout propagation patterns
//===----------------------------------------------------------------------===//

/// The pattern
///   onnx.Concat (zhigh.Unstick (%X1), zhigh.Unstick (%X2)) { axis })
/// can be replaced by
///   zhigh.Unstick (onnx.Concat (%X1, %X2) { new_axis })
class ONNXConcatLayoutPropagatePattern : public OpRewritePattern<ONNXConcatOp> {
public:
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const override {
    Operation *genericOp = concatOp.getOperation();
    Location loc = genericOp->getLoc();
    ValueRange inputs = concatOp.getInputs();
    IntegerAttr axis = concatOp.getAxisAttr();
    Value output = concatOp.getConcatResult();

    // Variables for capturing values and attributes used while creating ops
    StringAttr layout;
    bool allTheSame;
    std::tie(allTheSame, layout) =
        areProducedByUnstickOpSameLayout(rewriter, inputs);
    if (!allTheSame)
      return failure();

    // Only support LAYOUT_4D and LAYOUT_NHWC at this moment. They have the same
    // stickification scheme.
    if (!(isNHWCLayout(layout) || is4DLayout(layout)))
      return failure();

    if (!haveNoPadsWhenStickified(inputs, layout, axis))
      return failure();

    // Rewrite
    SmallVector<Value, 4> tblgen_repl_values;
    SmallVector<Value, 4> zTensors = getZTensors(rewriter, loc, inputs);
    IntegerAttr newAxis = getNewConcatAxis(rewriter, layout, axis);
    Type newOutputType = getZTensorType(rewriter, loc, output, layout);

    Value zOutput =
        rewriter.create<ONNXConcatOp>(loc, newOutputType, zTensors, newAxis);
    Value replacedValue =
        rewriter.create<ZHighUnstickOp>(loc, output.getType(), zOutput);
    rewriter.replaceOp(genericOp, replacedValue);
    return ::mlir::success();
  };

private:
  // Check if there are no pads along the given axis when stickifying values by
  // using the given layout.
  bool haveNoPadsWhenStickified(
      ValueRange values, StringAttr layoutAttr, IntegerAttr axisAttr) const {
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

    // C dimension is tiled by 64 when stickified. Hence, checking `C mod 64`
    // for padding.
    // TODO: get this info from affine_map that is used for stickiyfing NHWC.
    return llvm::all_of(values, [&layoutAttr](Value v) {
      if (mlir::isa<ShapedType>(v.getType()) &&
          mlir::cast<ShapedType>(v.getType()).hasRank()) {
        ArrayRef<int64_t> dims = mlir::cast<ShapedType>(v.getType()).getShape();
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

  IntegerAttr getNewConcatAxis(PatternRewriter &rewriter, StringAttr layout,
      IntegerAttr axisAttr) const {
    int axis = axisAttr.getValue().getSExtValue();
    if (isNHWCLayout(layout)) {
      SmallVector<int, 4> NCHWtoNHWC = {0, 3, 1, 2};
      return rewriter.getIntegerAttr(
          rewriter.getIntegerType(64, true), NCHWtoNHWC[axis]);
    }
    return axisAttr;
  }
};

/// Use anonymous namespace to avoid duplication symbol `populateWithGenerated`
/// among multiple tablegen-based definitions.

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Transform/ZHigh/ONNXZHighLayoutPropagation.inc"

//===----------------------------------------------------------------------===//
// ZHigh layout propagation Pass
//===----------------------------------------------------------------------===//

struct ZHighLayoutPropagationPass
    : public PassWrapper<ZHighLayoutPropagationPass,
          OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHighLayoutPropagationPass)

  StringRef getArgument() const override { return "zhigh-layout-prop"; }

  StringRef getDescription() const override {
    return "Layout propagation at ZHighIR.";
  }

  void runOnOperation() override {
    Operation *function = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // Layout propagation for ZHigh Ops.
    populateWithGenerated(patterns);

    // Concat
    patterns.insert<ONNXConcatLayoutPropagatePattern>(&getContext());

    // We want to canonicalize stick/unstick ops during this pass to simplify
    // rules in this pass.
    ZHighStickOp::getCanonicalizationPatterns(patterns, &getContext());
    ZHighUnstickOp::getCanonicalizationPatterns(patterns, &getContext());
    (void)applyPatternsGreedily(function, std::move(patterns));
  }
};
} // anonymous namespace

std::unique_ptr<Pass> createZHighLayoutPropagationPass() {
  return std::make_unique<ZHighLayoutPropagationPass>();
}

} // namespace zhigh
} // namespace onnx_mlir
