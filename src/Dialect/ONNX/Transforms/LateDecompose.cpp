/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- LateDecompose.cpp - Late Decomposition Patterns -----------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements late decomposition patterns for ONNX operations that
// were not handled by accelerators.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/LateDecompose.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

//===----------------------------------------------------------------------===//
// Pattern: Conv to Im2Col + MatMul + Reshape
//===----------------------------------------------------------------------===//

// Decompose non-1x1 convolutions into Im2Col + MatMul + Reshape.
// This transformation is applied to convolutions that:
// - Are 2D convolutions (rank = 4: N x C x H x W)
// - Have non-1x1 kernels (1x1 kernels are handled by ConvOpt)
// - Have group = 1 (grouped convolutions not supported yet)
// - Have shape information available
//
// Transformation:
//   Y = Conv(X, W, B)
// becomes:
//   X_col = Im2Col(X, kernel_shape, strides, pads, dilations)
//   W_2d = Reshape(W, [CO, CI*KH*KW])
//   Y_flat = MatMul(X_col, Transpose(W_2d))
//   if (hasBias):
//     Y_flat = Add(Y_flat, B)
//   Y = Reshape(Y_flat, [N, CO, OH, OW])

struct ConvToIm2ColPattern : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const final {
    // Check if this convolution should be decomposed.
    if (!shouldDecompose(convOp))
      return failure();

    Location loc = convOp.getLoc();
    Value X = convOp.getX();
    Value W = convOp.getW();
    Value B = convOp.getB();
    bool hasBias = !isNoneValue(B);

    // Create ONNX builder for cleaner code.
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

    // Get element types.
    ShapedType xType = mlir::cast<ShapedType>(X.getType());
    ShapedType wType = mlir::cast<ShapedType>(W.getType());
    
    auto wShape = wType.getShape();
    
    // Extract weight dimensions (these are always static).
    // W: [CO, CI, KH, KW]
    int64_t CO = wShape[0];
    int64_t CI = wShape[1];
    int64_t KH = wShape[2];
    int64_t KW = wShape[3];

    // Compute flattened kernel size: CI * KH * KW.
    int64_t kernelSize = CI * KH * KW;

    // Step 1: Create Im2Col operation.
    // Output: [?, CI*KH*KW] where ? is dynamic (N*OH*OW).
    SmallVector<int64_t, 2> im2colShape = {ShapedType::kDynamic, kernelSize};
    Type im2colOutputType =
        RankedTensorType::get(im2colShape, xType.getElementType());

    Value X_col = ONNXIm2ColOp::create(rewriter, loc,
        im2colOutputType, X,
        convOp.getAutoPadAttr(),
        convOp.getDilationsAttr(),
        convOp.getKernelShapeAttr(),
        convOp.getPadsAttr(),
        convOp.getStridesAttr());

    // Step 2: Reshape W from [CO, CI, KH, KW] to [CO, CI*KH*KW].

    // Create shape constant for reshape: [CO, kernelSize].
    SmallVector<int64_t, 2> w2dShape = {CO, kernelSize};
    Value shapeConst = create.onnx.constantInt64(w2dShape);

    // Infer output type for reshaped weight.
    Type w2dType = RankedTensorType::get(w2dShape, wType.getElementType());

    // Reshape weight tensor.
    Value W_2d = create.onnx.reshape(w2dType, W, shapeConst);

    // Step 3: Transpose W_2d to [kernelSize, CO] for MatMul.
    Value W_2d_T = create.onnx.transposeInt64(W_2d, {1, 0});

    // Step 4: MatMul: [?, CI*KH*KW] @ [CI*KH*KW, CO]
    //         Result: [?, CO] where ? is dynamic.
    SmallVector<int64_t, 2> matmulShape = {ShapedType::kDynamic, CO};
    Type matmulType =
        RankedTensorType::get(matmulShape, wType.getElementType());

    Value Y_flat = create.onnx.matmul(matmulType, X_col, W_2d_T);

    // Step 5: Add bias if present.
    Value Y_with_bias = Y_flat;
    if (hasBias) {
      // Bias shape is [CO], broadcasts to [N*OH*OW, CO].
      Y_with_bias = create.onnx.add(Y_flat, B);
    }

    // Step 6: Reshape back to [N, CO, OH, OW].
    // Compute output shape from input X and weight W using onnx.Dim.
    ShapedType convOutType = mlir::cast<ShapedType>(convOp.getType());
    
    // Extract individual dimensions using onnx.Dim.
    // From X: [N, CI, H, W] -> get N, H, W
    Value N = create.onnx.dim(X, 0);
    Value OH = create.onnx.dim(X, 2);  // Output height (same as input for SAME padding)
    Value OW = create.onnx.dim(X, 3);  // Output width (same as input for SAME padding)
    
    // From W: [CO, CI, KH, KW] -> get CO
    Value COVal = create.onnx.dim(W, 0);
    
    // Concatenate dimensions to form output shape: [N, CO, OH, OW]
    Type shapeType = RankedTensorType::get({4}, rewriter.getI64Type());
    Value outputShapeVals = create.onnx.concat(
        shapeType, {N, COVal, OH, OW}, 0);
    
    Value Y = create.onnx.reshape(convOutType, Y_with_bias, outputShapeVals);

    // Replace the original Conv with the final Reshape.
    rewriter.replaceOp(convOp, Y);

    return success();
  }

private:
  // Check if this convolution should be decomposed.
  bool shouldDecompose(ONNXConvOp convOp) const {
    // 1. Must have shape information.
    Value X = convOp.getX();
    Value W = convOp.getW();
    if (!hasShapeAndRank(X) || !hasShapeAndRank(W))
      return false;

    // 2. Weight tensor must have static shape (we need to know kernel dims).
    if (!hasStaticShape(W.getType()))
      return false;

    // 3. Must be 2D convolution (rank = 4: N x C x H x W).
    // For now, only support 2D convolutions.
    // Future: extend to 1D and 3D convolutions.
    ShapedType xType = mlir::cast<ShapedType>(X.getType());
    auto xShape = xType.getShape();
    int64_t rank = xShape.size();
    if (rank != 4)
      return false; // Only 2D convolutions for now.

    // 4. Group must be 1 (no grouped convolutions for now).
    if (convOp.getGroup() != 1)
      return false;

    // 5. Must NOT be 1x1 convolution (handled by ConvOpt).
    // For 2D convolution, check if kernel is [1, 1].
    ShapedType wType = mlir::cast<ShapedType>(W.getType());
    auto wShape = wType.getShape();
    int64_t KH = wShape[2];
    int64_t KW = wShape[3];

    if (KH == 1 && KW == 1)
      return false; // 1x1 kernel, let ConvOpt handle it.

    return true;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public Interface
//===----------------------------------------------------------------------===//

void getLateDecomposePatterns(RewritePatternSet &patterns) {
  patterns.add<ConvToIm2ColPattern>(patterns.getContext());
  // Future patterns can be added here.
}

namespace {

struct LateDecomposePass
    : public PassWrapper<LateDecomposePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LateDecomposePass)

  StringRef getArgument() const override { return "late-decompose"; }

  StringRef getDescription() const override {
    return "Decompose ONNX ops not handled by accelerators (Conv->Im2Col+"
           "MatMul, etc.)";
  }

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    getLateDecomposePatterns(patterns);

    if (failed(applyPatternsGreedily(function, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLateDecomposePass() {
  return std::make_unique<LateDecomposePass>();
}

} // namespace onnx_mlir

// Made with Bob
