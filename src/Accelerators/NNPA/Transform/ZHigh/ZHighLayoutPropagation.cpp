/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZHighLayoutPropagation.cpp - ZHigh High Level Optimizer ---===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
#include "src/Support/TypeUtilities.hpp"

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
  if (first.isa<BlockArgument>() || !isa<ZHighUnstickOp>(first.getDefiningOp()))
    return std::make_pair(false, nullptr);
  Value firstStickifiedVal = cast<ZHighUnstickOp>(first.getDefiningOp()).In();
  StringAttr firstLayout = convertZTensorDataLayoutToStringAttr(
      rewriter, getZTensorLayout(firstStickifiedVal.getType()));

  // Check all values.
  bool allTheSame = llvm::all_of(values, [&](Value v) {
    using namespace onnx_mlir::zhigh;
    if (v.isa<BlockArgument>() || !isa<ZHighUnstickOp>(v.getDefiningOp()))
      return false;
    Value stickifiedVal = cast<ZHighUnstickOp>(v.getDefiningOp()).In();
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
  ZHighStickOp stickOp = rewriter.create<ZHighStickOp>(loc, tensor, layout);
  (void)stickOp.inferShapes([](Region &region) {});

  Type returnType = stickOp.Out().getType();
  rewriter.eraseOp(stickOp);

  return returnType;
}

//===----------------------------------------------------------------------===//
// ZHigh layout propagation patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Layout propagation for ONNX unary element-wise operations
//
// ONNX unary operations, such as Relu and Tanh, do not depend on a specific
// layout. Thus, it is possible to use the layout of the previous zTensor. In
// other words, propagate the existing layout down to the unary operation, so
// that other rules can be applied to remove unstick-stick pairs.For example,
// This sequence of operations:
//   zhigh.Conv -> Unstick (NHWC) -> onnx.Relu
// will become:
//   zhigh.Conv -> onnx.Relu -> Unstick (NHWC)
// then, canonicalization of unstick/stick will remove the unstick-stick pair.
//
//===----------------------------------------------------------------------===//

template <typename ONNX_OP>
class ONNXUnaryOpLayoutPropPattern : public OpRewritePattern<ONNX_OP> {
public:
  using OpRewritePattern<ONNX_OP>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNX_OP unaryOp, PatternRewriter &rewriter) const override {
    Operation *genericOp = unaryOp.getOperation();
    Location loc = genericOp->getLoc();

    Value input = unaryOp.X();
    Value output = unaryOp.Y();

    // Input is a block argument, do nothing.
    if (input.dyn_cast<BlockArgument>())
      return failure();

    // Input is a CPU tensor, do nothing.
    auto unstickOp = dyn_cast<ZHighUnstickOp>(input.getDefiningOp());
    if (!unstickOp)
      return failure();

    // Input is unstickified from a zTensor. Do computation directly on the
    // zTensor.
    Value zTensor = unstickOp.In();
    Value zOutput = rewriter.create<ONNX_OP>(loc, zTensor.getType(), zTensor);
    Value replacedValue =
        rewriter.create<ZHighUnstickOp>(loc, output.getType(), zOutput);
    rewriter.replaceOp(genericOp, replacedValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Layout propagation for ONNX binary element-wise operations
//
// ONNX binary operations, such as Add and Div, do not depend on a specific
// layout. Thus, it is possible to use the layout of the previous zTensor. In
// other words, propagate the existing layout down to the binary operation, so
// that other rules can be applied to remove unstick-stick pairs.
//
// For example, this sequence of operations:
//   X -> Unstick (NHWC) -> onnx.Add
//                              ^
//                              |
//   Y -> Unstick (NHWC) --------
//
// will become:
//   X -> onnx.Add -> Unstick (NHWC)
//          ^
//          |
//   Y ------
//
// then, canonicalization of unstick/stick will remove the unstick-stick pairs.
//
// If X and Y has different layouts/ranks, e.g. onnx.Add is broadcasting. For
// example, X is a zTensor of layout 4D, Y is a normal tensor of rank 2:
//   X -> Unstick (4D) -> onnx.Add
//                              ^
//                              |
//              Y (2D) ----------
// will become:
//   X -> onnx.Add -> Unstick (4D)
//          ^
//          |
//   Y ------
// where the output of onnx.Add uses its rank to set layout, e.g. 1D for rank 1,
// 2D for rank 2, 3D for rank 3, and 4D for rank 4.
//
// Note: in case that one of the inputs is a normal tensor, only f32 is
// supported because dlfloat16 is converted to f32 while doing calculation.
//
//===----------------------------------------------------------------------===//

template <typename ONNX_OP>
class ONNXBinaryOpLayoutPropPattern : public OpRewritePattern<ONNX_OP> {
public:
  using OpRewritePattern<ONNX_OP>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNX_OP binaryOp, PatternRewriter &rewriter) const override {
    using DataLayout = ZTensorEncodingAttr::DataLayout;
    DataLayout NO_LAYOUT = ZTensorEncodingAttr::DataLayout::UNDEFINED;

    Operation *genericOp = binaryOp.getOperation();
    Location loc = genericOp->getLoc();

    Value A = binaryOp.A();
    Value B = binaryOp.B();
    Value output = binaryOp.C();

    // New inputs: at least one normal tensor comes from a zTensor.
    bool shouldRewrite = false;
    Value newA, newB;
    if (A.dyn_cast<BlockArgument>()) {
      newA = A;
    } else {
      if (auto unstickAOp = dyn_cast<ZHighUnstickOp>(A.getDefiningOp())) {
        newA = unstickAOp.In();
        shouldRewrite = true;
      } else {
        newA = A;
      }
    }
    if (B.dyn_cast<BlockArgument>()) {
      newB = B;
    } else {
      if (auto unstickBOp = dyn_cast<ZHighUnstickOp>(B.getDefiningOp())) {
        newB = unstickBOp.In();
        shouldRewrite = true;
      } else {
        newB = B;
      }
    }
    if (!shouldRewrite)
      return failure();

    // Get layout for the output zTensor.
    DataLayout outputLayout;
    DataLayout ALayout = getZTensorLayout(newA.getType());
    DataLayout BLayout = getZTensorLayout(newB.getType());
    // Both new inputs are a CPU or unranked tensor, do nothing.
    if ((ALayout == NO_LAYOUT) && (BLayout == NO_LAYOUT))
      return failure();
    // Only support f32 in the normal tensor.
    if (ALayout == NO_LAYOUT) {
      if (auto tensorType = llvm::dyn_cast<TensorType>(newA.getType())) {
        Type elementType = tensorType.getElementType();
        if (auto floatType = llvm::dyn_cast<FloatType>(elementType)) {
          if (floatType.getWidth() != 32)
            return failure();
        } else
          return failure();
      }
    }
    if (BLayout == NO_LAYOUT) {
      if (auto tensorType = llvm::dyn_cast<TensorType>(newB.getType())) {
        Type elementType = tensorType.getElementType();
        if (auto floatType = llvm::dyn_cast<FloatType>(elementType)) {
          if (floatType.getWidth() != 32)
            return failure();
        } else
          return failure();
      }
    }
    // TODO: Only support static dimensions in zTensor because access index in
    // case of broadcasting is not affine and memref normalization seems not
    // working well.
    //
    // If A and B have the same layout, use that layout for the output.
    // Otherwise, use the rank of the output to set a layout.
    if ((ALayout == BLayout) && (ALayout != NO_LAYOUT))
      outputLayout = ALayout;
    else {
      int64_t rank = getRank(output.getType());
      outputLayout = getZTensorDataLayoutByRank(rank);
    }

    // Construct the zTensor type from the output CPU tensor.
    Type zOutputType = getZTensorType(rewriter, loc, output,
        convertZTensorDataLayoutToStringAttr(rewriter, outputLayout));

    Value zOutput = rewriter.create<ONNX_OP>(loc, zOutputType, newA, newB);
    Value replacedValue =
        rewriter.create<ZHighUnstickOp>(loc, output.getType(), zOutput);
    rewriter.replaceOp(genericOp, replacedValue);
    return success();
  }
};

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
    ValueRange inputs = concatOp.inputs();
    IntegerAttr axis = concatOp.axisAttr();
    Value output = concatOp.concat_result();

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
    func::FuncOp function = getOperation();
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    RewritePatternSet patterns(ctx);

    // Layout propagation for ZHigh Ops.
    populateWithGenerated(patterns);

    // Layout propagation for ONNX Ops. Since DLFLOAT16 is not supported in
    // LLVM, we disable computation operators like Add, Div, etc. and enable
    // only data movement operators like Concat.

    // Add
    patterns.insert<ONNXBinaryOpLayoutPropPattern<ONNXAddOp>>(ctx);
    // Div
    patterns.insert<ONNXBinaryOpLayoutPropPattern<ONNXDivOp>>(ctx);
    // Mul
    patterns.insert<ONNXBinaryOpLayoutPropPattern<ONNXMulOp>>(ctx);
    // Sub
    patterns.insert<ONNXBinaryOpLayoutPropPattern<ONNXSubOp>>(ctx);
    // Reciprocal
    patterns.insert<ONNXUnaryOpLayoutPropPattern<ONNXReciprocalOp>>(ctx);
    // Sqrt
    patterns.insert<ONNXUnaryOpLayoutPropPattern<ONNXSqrtOp>>(ctx);

    // Concat
    patterns.insert<ONNXConcatLayoutPropagatePattern>(ctx);

    // We want to canonicalize stick/unstick ops during this pass to simplify
    // rules in this pass.
    ZHighStickOp::getCanonicalizationPatterns(patterns, ctx);
    ZHighUnstickOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};
} // anonymous namespace

std::unique_ptr<Pass> createZHighLayoutPropagationPass() {
  return std::make_unique<ZHighLayoutPropagationPass>();
}

} // namespace zhigh
} // namespace onnx_mlir
