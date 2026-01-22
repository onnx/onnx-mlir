// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "llvm/ADT/SmallVector.h"

#include <vector>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Helper function to expand dilated weights for NCHW layout
// Input weights shape: [out_channels, in_channels, kH, kW] (NCHW format)
// Output weights shape: [out_channels, in_channels, kH', kW'] where kH' and kW' are expanded
mlir::DenseElementsAttr expandDilatedWeightsNCHW(
    mlir::DenseElementsAttr weightsAttr, llvm::ArrayRef<int64_t> originalShape,
    int64_t dilation) {
  // Original shape: [out_channels, in_channels, kH, kW] (NCHW)
  int64_t out_channels = originalShape[0];
  int64_t in_channels = originalShape[1];
  int64_t org_h = originalShape[2]; // Kernel height
  int64_t org_w = originalShape[3]; // Kernel width

  // New dimensions with dilation
  int64_t new_h = org_h + (org_h - 1) * (dilation - 1);
  int64_t new_w = org_w + (org_w - 1) * (dilation - 1);

  // Extract original weights as float
  auto weightsValues = weightsAttr.getValues<float>();
  std::vector<float> originalWeights(weightsValues.begin(),
                                     weightsValues.end());

  // Create expanded weights initialized to zero
  std::vector<float> expandedWeights(out_channels * in_channels * new_h * new_w,
                                     0.0f);

  // Copy weights with dilation spacing
  // Layout: [out_channels, in_channels, kH, kW]
  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t ic = 0; ic < in_channels; ++ic) {
      for (int64_t kh = 0; kh < new_h; ++kh) {
        for (int64_t kw = 0; kw < new_w; ++kw) {
          // Only copy at positions divisible by dilation
          if ((kh % dilation == 0) && (kw % dilation == 0)) {
            // Memory layout: [out_channels][in_channels][kH][kW]
            int64_t src_idx = oc * in_channels * org_h * org_w +
                              ic * org_h * org_w + (kh / dilation) * org_w +
                              (kw / dilation);
            int64_t dst_idx = oc * in_channels * new_h * new_w +
                              ic * new_h * new_w + kh * new_w + kw;
            expandedWeights[dst_idx] = originalWeights[src_idx];
          }
        }
      }
    }
  }

  // Create new shape and tensor (NCHW)
  SmallVector<int64_t> newShape = {out_channels, in_channels, new_h, new_w};
  auto tensorType =
      RankedTensorType::get(newShape, weightsAttr.getElementType());
  return mlir::DenseElementsAttr::get(tensorType,
                                      llvm::ArrayRef(expandedWeights));
}

//===----------------------------------------------------------------------===//
// Pattern: RemoveDilationConv
//===----------------------------------------------------------------------===//

struct RemoveDilationConv : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXConvOp convOp,
                                PatternRewriter &rewriter) const override {
  // Verify it's a 2D convolution (weights are 4D)
  auto weightsType = mlir::dyn_cast<RankedTensorType>(convOp.getW().getType());
  if (!weightsType || weightsType.getRank() != 4)
    return failure();

  // Check dilation attribute
  auto dilationsAttr = convOp.getDilationsAttr();
  if (!dilationsAttr)
    return failure();

  llvm::SmallVector<int64_t> dilations;
  for (auto attr : dilationsAttr) {
    dilations.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  }

  if (dilations.size() != 2)
    return failure();

  // Must have uniform non-unity dilation (2, 3, or 4)
  int64_t dilation = dilations[0];
  if (dilation != dilations[1] || dilation < 2 || dilation > 4)
    return failure();

  // Get weights input
  Value weightsInput = convOp.getW();

  // Weights must be a constant
  auto weightsConst = weightsInput.getDefiningOp<ONNXConstantOp>();
  if (!weightsConst)
    return failure();

  // ============== Pattern matched! Now perform transformation ==============
  Location loc = convOp.getLoc();

  // Get original weights
  auto originalWeightsAttr =
      mlir::dyn_cast<mlir::DenseElementsAttr>(weightsConst.getValueAttr());

  if (!originalWeightsAttr)
    return failure();

  auto originalShape = mlir::cast<ShapedType>(weightsConst.getType()).getShape();

  // Validate weights shape (must be 4D and square kernel)
  // For NCHW: [out_channels, in_channels, kH, kW] where kH==kW
  if (originalShape.size() != 4 || originalShape[2] != originalShape[3])
    return failure();

  int64_t org_kernel = originalShape[2];

  // Calculate new kernel size
  int64_t new_kernel = org_kernel + (org_kernel - 1) * (dilation - 1);

  // Expand weights with dilation (NCHW layout)
  auto expandedWeightsAttr =
      expandDilatedWeightsNCHW(originalWeightsAttr, originalShape, dilation);

  auto valueAttr = rewriter.getNamedAttr("value", expandedWeightsAttr);

  // Create new constant with expanded weights [out_channels, in_channels,
  // new_h, new_w]
  auto newWeightsConst = rewriter.create<ONNXConstantOp>(
      loc, expandedWeightsAttr.getType(), mlir::ValueRange{},
      mlir::ArrayRef<mlir::NamedAttribute>{valueAttr});

  // Get the output type from original conv to ensure shape compatibility
  auto originalOutputType =
      mlir::cast<RankedTensorType>(convOp.getResult().getType());

  // Create new Conv with dilation=1 and expanded kernel
  SmallVector<int64_t> newDilations = {1, 1};
  SmallVector<int64_t> newKernelShape = {new_kernel, new_kernel};

  // Create new Conv operation
  rewriter.replaceOpWithNewOp<ONNXConvOp>(
      convOp,
      originalOutputType, // Use original output type for shape inference
      convOp.getX(),      // Keep original input
      newWeightsConst.getResult(),              // New expanded weights
      convOp.getB(),                            // Keep bias if present
      convOp.getAutoPadAttr(),                  // Keep auto_pad setting
      rewriter.getI64ArrayAttr(newDilations),   // dilation = [1, 1]
      convOp.getGroupAttr(),                    // Keep group setting
      rewriter.getI64ArrayAttr(newKernelShape), // new kernel size
      convOp.getPadsAttr(),                     // adjusted pads
      convOp.getStridesAttr()                   // Keep strides
  );

  return success();
}
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct RemoveDilationConvPass
    : public PassWrapper<RemoveDilationConvPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "remove-dilation-conv"; }
  StringRef getDescription() const override {
    return "Replace dilated convolutions with standard convolutions using expanded kernels";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RemoveDilationConv>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createRemoveDilationConvPass() {
  return std::make_unique<RemoveDilationConvPass>();
}

} // namespace onnx_mlir
