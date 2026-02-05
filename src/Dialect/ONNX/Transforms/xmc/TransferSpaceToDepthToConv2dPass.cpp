// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"

#include <numeric>
#include <vector>

using namespace mlir;

namespace {

/// Create an onnx.Constant with a given result type and `value` attribute.
/// This is used for quantized tensors where the attribute element type may be
/// the storage type (e.g. i8) while the result type is `!quant.uniform<...>`.
static Value createOnnxConstant(PatternRewriter &rewriter, Location loc,
    Type resultType, Attribute valueAttr) {
  OperationState st(loc, "onnx.Constant");
  st.addTypes(resultType);
  st.addAttribute("value", valueAttr);
  Operation *op = rewriter.create(st);
  return op->getResult(0);
}

/// Transfer a very specific SpaceToDepth pattern to a Conv2D.
/// Note: this mirrors the existing FlexML XCompiler behavior.
struct TransferSpaceToDepthToConv2dPattern : public RewritePattern {
  TransferSpaceToDepthToConv2dPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    // Match SpaceToDepth op.
    if (op->getName().getStringRef() != "onnx.SpaceToDepth")
      return failure();
    if (op->getNumOperands() < 1 || op->getNumResults() < 1)
      return failure();

    Value spaceToDepthInput = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(spaceToDepthInput.getType());
    if (!inputType || !inputType.hasStaticShape())
      return failure();

    ArrayRef<int64_t> inputShape = inputType.getShape();

    // Model-specific guard: expect [1, 3, 512, 512] in NCHW.
    if (inputShape.size() != 4 || inputShape[0] != 1 || inputShape[1] != 3 ||
        inputShape[2] != 512 || inputShape[3] != 512)
      return failure();

    // blocksize attribute.
    auto blockSizeAttr =
        dyn_cast_or_null<IntegerAttr>(op->getAttr("blocksize"));
    if (!blockSizeAttr)
      return failure();
    int64_t blockSize = blockSizeAttr.getValue().getSExtValue();
    if (blockSize != 4)
      return failure();

    auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!outputType)
      return failure();

    Location loc = op->getLoc();
    int64_t inputChannels = inputShape[1]; // NCHW

    // Weight shape: [out_channels, in_channels, kH, kW]
    // out_channels = in_channels * block_size * block_size
    SmallVector<int64_t, 4> weightShape = {
        inputChannels * blockSize * blockSize, inputChannels, blockSize,
        blockSize};
    int64_t weightsSize = std::accumulate(weightShape.begin(),
        weightShape.end(), 1LL, std::multiplies<int64_t>());

    // Identity-like weights (i8).
    std::vector<int8_t> weightsData(weightsSize, 0);
    for (int64_t kh = 0; kh < blockSize; ++kh) {
      for (int64_t kw = 0; kw < blockSize; ++kw) {
        for (int64_t ic = 0; ic < inputChannels; ++ic) {
          int64_t oc = (kh * blockSize + kw) * inputChannels + ic;
          int64_t idx = oc * inputChannels * blockSize * blockSize +
                        ic * blockSize * blockSize + kh * blockSize + kw;
          weightsData[idx] = 1;
        }
      }
    }

    // Quantized weight type.
    auto weightQuantType = quant::UniformQuantizedType::get(
        /*flags=*/quant::QuantizationFlags::Signed,
        /*storageType=*/rewriter.getI8Type(),
        /*expressedType=*/rewriter.getF32Type(),
        /*scale=*/1.0,
        /*zeroPoint=*/0,
        /*storageTypeMin=*/-128,
        /*storageTypeMax=*/127);
    auto weightTensorType = RankedTensorType::get(weightShape, weightQuantType);
    auto weightAttrType =
        RankedTensorType::get(weightShape, rewriter.getI8Type());
    auto weightAttr = DenseIntElementsAttr::get(
        weightAttrType, llvm::ArrayRef<int8_t>(weightsData));
    Value weightConst =
        createOnnxConstant(rewriter, loc, weightTensorType, weightAttr);

    // Bias constant (i8, all zeros).
    int64_t numOutputChannels = weightShape[0];
    SmallVector<int64_t, 1> biasShape = {numOutputChannels};
    std::vector<int8_t> biasData(numOutputChannels, 0);

    // Bias scale = input_scale * weight_scale (weight_scale = 1.0).
    double inputScaleValue = 1.0;
    if (auto qType =
            dyn_cast<quant::UniformQuantizedType>(inputType.getElementType()))
      inputScaleValue = qType.getScale();
    float biasScale = static_cast<float>(inputScaleValue);

    auto biasQuantType = quant::UniformQuantizedType::get(
        /*flags=*/quant::QuantizationFlags::Signed,
        /*storageType=*/rewriter.getI8Type(),
        /*expressedType=*/rewriter.getF32Type(),
        /*scale=*/biasScale,
        /*zeroPoint=*/0,
        /*storageTypeMin=*/-128,
        /*storageTypeMax=*/127);
    auto biasTensorType = RankedTensorType::get(biasShape, biasQuantType);
    auto biasAttrType = RankedTensorType::get(biasShape, rewriter.getI8Type());
    auto biasAttr = DenseIntElementsAttr::get(
        biasAttrType, llvm::ArrayRef<int8_t>(biasData));
    Value biasConst =
        createOnnxConstant(rewriter, loc, biasTensorType, biasAttr);

    // Conv attributes.
    SmallVector<int64_t, 2> kernel = {blockSize, blockSize};
    SmallVector<int64_t, 2> strides = {blockSize, blockSize};
    SmallVector<int64_t, 4> pads = {0, 0, 0, 0};
    SmallVector<int64_t, 2> dilations = {1, 1};

    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value convResult = onnxBuilder.conv(outputType, spaceToDepthInput,
        weightConst, biasConst, /*autoPad=*/"NOTSET", dilations, /*group=*/1,
        kernel, pads, strides);

    rewriter.replaceOp(op, convResult);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct TransferSpaceToDepthToConv2dPass
    : public PassWrapper<TransferSpaceToDepthToConv2dPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transfer-space-to-depth-to-conv2d";
  }
  StringRef getDescription() const override {
    return "Transfer (specific) SpaceToDepth patterns to Conv2D";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<TransferSpaceToDepthToConv2dPattern>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createTransferSpaceToDepthToConv2dPass() {
  return std::make_unique<TransferSpaceToDepthToConv2dPass>();
}

} // namespace onnx_mlir
