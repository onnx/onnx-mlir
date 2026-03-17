// Copyright (C) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass splits depthwise conv2d operations with channel multiplier > 1
// into multiple depthwise convs with channel_multiplier=1, followed by concat.
// This is a translation of the XIR
// TransferDepthwiseConv2dwithChannelMultiplierPass to MLIR.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "transfer-depthwise-conv2d-channel-multiplier"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if a conv op is a depthwise convolution (group == input_channels)
/// and has channel_multiplier > 1
template <typename ConvOpT>
bool isDepthwiseConvWithChannelMultiplier(
    ConvOpT convOp, int64_t &channelMultiplier, int64_t &inputChannels) {
  auto inputType = dyn_cast<RankedTensorType>(convOp.getX().getType());
  auto weightType = dyn_cast<RankedTensorType>(convOp.getW().getType());

  if (!inputType || !weightType || !inputType.hasStaticShape() ||
      !weightType.hasStaticShape()) {
    return false;
  }

  auto inputShape = inputType.getShape();
  auto weightShape = weightType.getShape();

  // Weight shape for Conv: [M, C/group, kH, kW] (NCHW) or [M, kH, kW, C/group]
  // (NHWC for XFE) For standard ONNX Conv: [M, C/group, kH, kW]
  if (weightShape.size() != 4) {
    return false;
  }

  int64_t group = convOp.getGroup();
  int64_t outputChannels = weightShape[0]; // M

  // For depthwise conv: group == input_channels and C/group == 1
  // Input shape for NCHW: [N, C, H, W] - C is at index 1
  // Input shape for NHWC: [N, H, W, C] - C is at index 3
  int64_t inputChannelDim;
  if constexpr (std::is_same_v<ConvOpT, ONNXConvOp>) {
    // NCHW format
    inputChannelDim = inputShape[1];
  } else {
    // XFE ops use NHWC format
    inputChannelDim = inputShape[3];
  }

  // Check if this is a depthwise conv (group == input_channels)
  if (group != inputChannelDim) {
    return false;
  }

  // Safety check: output channels must be a multiple of group
  if (outputChannels % group != 0) {
    return false;
  }

  // Calculate channel multiplier = output_channels / group
  channelMultiplier = outputChannels / group;
  inputChannels = inputChannelDim;

  // Only handle channel_multiplier > 1
  return channelMultiplier > 1;
}

/// Extract values from a DenseElementsAttr
template <typename T>
SmallVector<T> extractDenseValues(DenseElementsAttr attr) {
  SmallVector<T> result;
  if (!attr)
    return result;

  for (auto val : attr.getValues<T>()) {
    result.push_back(val);
  }
  return result;
}

/// Extract and split weights for a given channel multiplier index
/// Returns the split weights for the specified cmIdx
template <typename T>
SmallVector<T> extractSplitWeights(DenseElementsAttr denseWeightAttr,
    int64_t cmIdx, int64_t inputChannels, int64_t weightsPerChannel) {
  SmallVector<T> splitWeights;
  splitWeights.reserve(inputChannels * weightsPerChannel);

  auto allWeights = extractDenseValues<T>(denseWeightAttr);
  for (int64_t c = 0; c < inputChannels; c++) {
    int64_t srcIdx = (cmIdx * inputChannels + c) * weightsPerChannel;
    for (int64_t i = 0; i < weightsPerChannel; i++) {
      splitWeights.push_back(allWeights[srcIdx + i]);
    }
  }
  return splitWeights;
}

/// Extract and split bias for a given channel multiplier index
template <typename T>
SmallVector<T> extractSplitBias(
    DenseElementsAttr denseBiasAttr, int64_t cmIdx, int64_t inputChannels) {
  SmallVector<T> biasValues;
  biasValues.reserve(inputChannels);

  auto allBias = extractDenseValues<T>(denseBiasAttr);
  for (int64_t c = 0; c < inputChannels; c++) {
    int64_t srcIdx = cmIdx * inputChannels + c;
    biasValues.push_back(allBias[srcIdx]);
  }
  return biasValues;
}

/// Create a constant op with the given values and shape (templatized for
/// different types)
template <typename T>
Value createConstant(PatternRewriter &rewriter, Location loc,
    ArrayRef<T> values, ArrayRef<int64_t> shape, Type elementType) {
  auto tensorType = RankedTensorType::get(shape, elementType);
  DenseElementsAttr attr;

  if constexpr (std::is_floating_point_v<T>) {
    attr = DenseFPElementsAttr::get(tensorType, values);
  } else {
    attr = DenseIntElementsAttr::get(tensorType, values);
  }

  return rewriter.create<ONNXConstantOp>(loc, tensorType, Attribute(), attr,
      FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(),
      ArrayAttr());
}

/// Create split weight constant based on element type
Value createSplitWeightConstant(PatternRewriter &rewriter, Location loc,
    DenseElementsAttr denseWeightAttr, int64_t cmIdx, int64_t inputChannels,
    int64_t weightsPerChannel, ArrayRef<int64_t> splitWeightShape) {
  Type elementType = denseWeightAttr.getElementType();

  if (elementType.isF32()) {
    auto splitWeights = extractSplitWeights<float>(
        denseWeightAttr, cmIdx, inputChannels, weightsPerChannel);
    return createConstant<float>(
        rewriter, loc, splitWeights, splitWeightShape, elementType);
  } else if (elementType.isF64()) {
    auto splitWeights = extractSplitWeights<double>(
        denseWeightAttr, cmIdx, inputChannels, weightsPerChannel);
    return createConstant<double>(
        rewriter, loc, splitWeights, splitWeightShape, elementType);
  } else if (elementType.isSignedInteger(8)) {
    auto splitWeights = extractSplitWeights<int8_t>(
        denseWeightAttr, cmIdx, inputChannels, weightsPerChannel);
    return createConstant<int8_t>(
        rewriter, loc, splitWeights, splitWeightShape, elementType);
  } else if (elementType.isUnsignedInteger(8)) {
    auto splitWeights = extractSplitWeights<uint8_t>(
        denseWeightAttr, cmIdx, inputChannels, weightsPerChannel);
    return createConstant<uint8_t>(
        rewriter, loc, splitWeights, splitWeightShape, elementType);
  } else if (elementType.isSignedInteger(16)) {
    auto splitWeights = extractSplitWeights<int16_t>(
        denseWeightAttr, cmIdx, inputChannels, weightsPerChannel);
    return createConstant<int16_t>(
        rewriter, loc, splitWeights, splitWeightShape, elementType);
  } else if (elementType.isUnsignedInteger(16)) {
    auto splitWeights = extractSplitWeights<uint16_t>(
        denseWeightAttr, cmIdx, inputChannels, weightsPerChannel);
    return createConstant<uint16_t>(
        rewriter, loc, splitWeights, splitWeightShape, elementType);
  }

  return Value(); // Unsupported type
}

/// Create split bias constant based on element type
Value createSplitBiasConstant(PatternRewriter &rewriter, Location loc,
    DenseElementsAttr denseBiasAttr, int64_t cmIdx, int64_t inputChannels) {
  Type elementType = denseBiasAttr.getElementType();
  SmallVector<int64_t> biasShape = {inputChannels};

  if (elementType.isF32()) {
    auto biasValues =
        extractSplitBias<float>(denseBiasAttr, cmIdx, inputChannels);
    return createConstant<float>(
        rewriter, loc, biasValues, biasShape, elementType);
  } else if (elementType.isF64()) {
    auto biasValues =
        extractSplitBias<double>(denseBiasAttr, cmIdx, inputChannels);
    return createConstant<double>(
        rewriter, loc, biasValues, biasShape, elementType);
  } else if (elementType.isSignedInteger(8)) {
    auto biasValues =
        extractSplitBias<int8_t>(denseBiasAttr, cmIdx, inputChannels);
    return createConstant<int8_t>(
        rewriter, loc, biasValues, biasShape, elementType);
  } else if (elementType.isUnsignedInteger(8)) {
    auto biasValues =
        extractSplitBias<uint8_t>(denseBiasAttr, cmIdx, inputChannels);
    return createConstant<uint8_t>(
        rewriter, loc, biasValues, biasShape, elementType);
  } else if (elementType.isSignedInteger(16)) {
    auto biasValues =
        extractSplitBias<int16_t>(denseBiasAttr, cmIdx, inputChannels);
    return createConstant<int16_t>(
        rewriter, loc, biasValues, biasShape, elementType);
  } else if (elementType.isUnsignedInteger(16)) {
    auto biasValues =
        extractSplitBias<uint16_t>(denseBiasAttr, cmIdx, inputChannels);
    return createConstant<uint16_t>(
        rewriter, loc, biasValues, biasShape, elementType);
  } else if (elementType.isInteger(32)) {
    auto biasValues =
        extractSplitBias<int32_t>(denseBiasAttr, cmIdx, inputChannels);
    return createConstant<int32_t>(
        rewriter, loc, biasValues, biasShape, elementType);
  }

  return Value(); // Unsupported type
}

//===----------------------------------------------------------------------===//
// Pattern: Split Depthwise Conv with Channel Multiplier (ONNX Conv)
//===----------------------------------------------------------------------===//

struct SplitDepthwiseConvPattern : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();

    int64_t channelMultiplier = 0;
    int64_t inputChannels = 0;

    if (!isDepthwiseConvWithChannelMultiplier(
            convOp, channelMultiplier, inputChannels)) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Found depthwise conv with channel_multiplier="
                            << channelMultiplier
                            << ", input_channels=" << inputChannels << "\n");

    auto weightType = cast<RankedTensorType>(convOp.getW().getType());
    auto outputType = cast<RankedTensorType>(convOp.getType());
    auto weightShape = weightType.getShape();
    auto outputShape = outputType.getShape();

    // Get weight constant
    auto weightConstOp = convOp.getW().getDefiningOp<ONNXConstantOp>();
    if (!weightConstOp) {
      LLVM_DEBUG(llvm::dbgs() << "Weight is not a constant, skipping\n");
      return failure();
    }

    auto weightAttr = weightConstOp.getValueAttr();
    if (!weightAttr) {
      return failure();
    }

    auto denseWeightAttr = dyn_cast<DenseElementsAttr>(weightAttr);
    if (!denseWeightAttr) {
      return failure();
    }

    // Weight shape: [M, 1, kH, kW] where M = channelMultiplier * inputChannels
    int64_t kH = weightShape[2];
    int64_t kW = weightShape[3];
    int64_t weightsPerChannel = kH * kW; // Since C/group = 1

    // Bounds check: validate weight tensor size
    int64_t expectedWeightElements =
        channelMultiplier * inputChannels * weightsPerChannel;
    if (static_cast<int64_t>(denseWeightAttr.getNumElements()) !=
        expectedWeightElements) {
      LLVM_DEBUG(llvm::dbgs() << "Weight tensor size mismatch: expected "
                              << expectedWeightElements << ", got "
                              << denseWeightAttr.getNumElements() << "\n");
      return failure();
    }

    // Get bias if present
    Value origBias = convOp.getB();
    bool hasBias = !isa<NoneType>(origBias.getType());
    DenseElementsAttr denseBiasAttr;
    if (hasBias) {
      auto biasConstOp = origBias.getDefiningOp<ONNXConstantOp>();
      if (biasConstOp) {
        auto biasAttr = biasConstOp.getValueAttr();
        denseBiasAttr = dyn_cast<DenseElementsAttr>(biasAttr);
      }
    }

    // Bounds check: validate bias tensor size if present
    if (hasBias && denseBiasAttr) {
      int64_t expectedBiasElements = channelMultiplier * inputChannels;
      if (static_cast<int64_t>(denseBiasAttr.getNumElements()) !=
          expectedBiasElements) {
        LLVM_DEBUG(llvm::dbgs() << "Bias tensor size mismatch: expected "
                                << expectedBiasElements << ", got "
                                << denseBiasAttr.getNumElements() << "\n");
        return failure();
      }
    }

    // Create channelMultiplier separate convolutions
    SmallVector<Value> concatInputs;
    concatInputs.reserve(channelMultiplier);

    // Calculate output shape for each split conv
    // Original output: [N, M, outH, outW] where M = channelMultiplier *
    // inputChannels New output per conv: [N, inputChannels, outH, outW]
    SmallVector<int64_t> splitOutputShape = {
        outputShape[0], inputChannels, outputShape[2], outputShape[3]};
    auto splitOutputType =
        RankedTensorType::get(splitOutputShape, outputType.getElementType());

    // New weight shape for each split: [inputChannels, 1, kH, kW]
    SmallVector<int64_t> splitWeightShape = {inputChannels, 1, kH, kW};

    // Create OnnxBuilder for utility functions
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);

    for (int64_t cmIdx = 0; cmIdx < channelMultiplier; cmIdx++) {
      // Create split weight constant using templatized helper
      Value splitWeight =
          createSplitWeightConstant(rewriter, loc, denseWeightAttr, cmIdx,
              inputChannels, weightsPerChannel, splitWeightShape);
      if (!splitWeight) {
        LLVM_DEBUG(
            llvm::dbgs() << "Unsupported weight element type, skipping\n");
        return failure();
      }

      // Create split bias if present
      Value splitBias;
      if (hasBias && denseBiasAttr) {
        splitBias = createSplitBiasConstant(
            rewriter, loc, denseBiasAttr, cmIdx, inputChannels);
        if (!splitBias) {
          LLVM_DEBUG(
              llvm::dbgs() << "Unsupported bias element type, skipping\n");
          return failure();
        }
      } else {
        splitBias = onnxBuilder.none();
      }

      // Create split depthwise conv with channel_multiplier = 1
      // group stays the same (= inputChannels)
      auto splitConvOp = rewriter.create<ONNXConvOp>(loc, splitOutputType,
          convOp.getX(), splitWeight, splitBias, convOp.getAutoPadAttr(),
          convOp.getDilationsAttr(), convOp.getGroupAttr(),
          convOp.getKernelShapeAttr(), convOp.getPadsAttr(),
          convOp.getStridesAttr());

      concatInputs.push_back(splitConvOp.getResult());

      LLVM_DEBUG(llvm::dbgs()
                 << "Created split conv " << cmIdx << " with output shape ["
                 << splitOutputShape[0] << ", " << splitOutputShape[1] << ", "
                 << splitOutputShape[2] << ", " << splitOutputShape[3]
                 << "]\n");
    }

    // Create concat to combine outputs along channel dimension (axis=1 for
    // NCHW) Create signed i64 type for ONNX attributes (si64)
    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);
    auto concatOp = rewriter.create<ONNXConcatOp>(
        loc, outputType, concatInputs, IntegerAttr::get(si64Type, 1));

    rewriter.replaceOp(convOp, concatOp.getResult());

    LLVM_DEBUG(llvm::dbgs()
               << "Successfully split depthwise conv with channel_multiplier="
               << channelMultiplier << " into " << channelMultiplier
               << " convs + concat\n");

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Split Depthwise Conv with Channel Multiplier (XFE Conv - NHWC)
//===----------------------------------------------------------------------===//

struct SplitXFEDepthwiseConvPattern : public OpRewritePattern<XFEConvOp> {
  using OpRewritePattern<XFEConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      XFEConvOp convOp, PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();

    int64_t channelMultiplier = 0;
    int64_t inputChannels = 0;

    if (!isDepthwiseConvWithChannelMultiplier(
            convOp, channelMultiplier, inputChannels)) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Found XFE depthwise conv with channel_multiplier="
               << channelMultiplier << ", input_channels=" << inputChannels
               << "\n");

    auto weightType = cast<RankedTensorType>(convOp.getW().getType());
    auto outputType = cast<RankedTensorType>(convOp.getY().getType());
    auto weightShape = weightType.getShape();
    auto outputShape = outputType.getShape();

    // Get weight constant
    auto weightConstOp = convOp.getW().getDefiningOp<ONNXConstantOp>();
    if (!weightConstOp) {
      LLVM_DEBUG(llvm::dbgs() << "Weight is not a constant, skipping\n");
      return failure();
    }

    auto weightAttr = weightConstOp.getValueAttr();
    if (!weightAttr) {
      return failure();
    }

    auto denseWeightAttr = dyn_cast<DenseElementsAttr>(weightAttr);
    if (!denseWeightAttr) {
      return failure();
    }

    // XFE Conv weight shape: [M, 1, kH, kW] (still NCHW for weights)
    int64_t kH = weightShape[2];
    int64_t kW = weightShape[3];
    int64_t weightsPerChannel = kH * kW;

    // Bounds check: validate weight tensor size
    int64_t expectedWeightElements =
        channelMultiplier * inputChannels * weightsPerChannel;
    if (static_cast<int64_t>(denseWeightAttr.getNumElements()) !=
        expectedWeightElements) {
      LLVM_DEBUG(llvm::dbgs() << "Weight tensor size mismatch: expected "
                              << expectedWeightElements << ", got "
                              << denseWeightAttr.getNumElements() << "\n");
      return failure();
    }

    // Get bias if present
    Value origBias = convOp.getB();
    bool hasBias = !isa<NoneType>(origBias.getType());
    DenseElementsAttr denseBiasAttr;
    if (hasBias) {
      auto biasConstOp = origBias.getDefiningOp<ONNXConstantOp>();
      if (biasConstOp) {
        auto biasAttr = biasConstOp.getValueAttr();
        denseBiasAttr = dyn_cast<DenseElementsAttr>(biasAttr);
      }
    }

    // Bounds check: validate bias tensor size if present
    if (hasBias && denseBiasAttr) {
      int64_t expectedBiasElements = channelMultiplier * inputChannels;
      if (static_cast<int64_t>(denseBiasAttr.getNumElements()) !=
          expectedBiasElements) {
        LLVM_DEBUG(llvm::dbgs() << "Bias tensor size mismatch: expected "
                                << expectedBiasElements << ", got "
                                << denseBiasAttr.getNumElements() << "\n");
        return failure();
      }
    }

    // Create channelMultiplier separate convolutions
    SmallVector<Value> concatInputs;
    concatInputs.reserve(channelMultiplier);

    // XFE output shape is NHWC: [N, outH, outW, M]
    // Split output shape: [N, outH, outW, inputChannels]
    SmallVector<int64_t> splitOutputShape = {
        outputShape[0], outputShape[1], outputShape[2], inputChannels};
    auto splitOutputType =
        RankedTensorType::get(splitOutputShape, outputType.getElementType());

    // New weight shape for each split: [inputChannels, 1, kH, kW]
    SmallVector<int64_t> splitWeightShape = {inputChannels, 1, kH, kW};

    // Create OnnxBuilder for utility functions
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);

    for (int64_t cmIdx = 0; cmIdx < channelMultiplier; cmIdx++) {
      // Create split weight constant using templatized helper
      Value splitWeight =
          createSplitWeightConstant(rewriter, loc, denseWeightAttr, cmIdx,
              inputChannels, weightsPerChannel, splitWeightShape);
      if (!splitWeight) {
        LLVM_DEBUG(
            llvm::dbgs() << "Unsupported weight element type, skipping\n");
        return failure();
      }

      // Create split bias if present
      Value splitBias;
      if (hasBias && denseBiasAttr) {
        splitBias = createSplitBiasConstant(
            rewriter, loc, denseBiasAttr, cmIdx, inputChannels);
        if (!splitBias) {
          LLVM_DEBUG(
              llvm::dbgs() << "Unsupported bias element type, skipping\n");
          return failure();
        }
      } else {
        splitBias = onnxBuilder.none();
      }

      // Create split XFE depthwise conv
      auto splitConvOp = rewriter.create<XFEConvOp>(loc, splitOutputType,
          convOp.getX(), splitWeight, splitBias, convOp.getActivationAttr(),
          convOp.getAutoPadAttr(), convOp.getDilationsAttr(),
          convOp.getGroupAttr(), convOp.getKernelShapeAttr(),
          convOp.getPadsAttr(), convOp.getStridesAttr());

      concatInputs.push_back(splitConvOp.getResult());

      LLVM_DEBUG(llvm::dbgs()
                 << "Created split XFE conv " << cmIdx << " with output shape ["
                 << splitOutputShape[0] << ", " << splitOutputShape[1] << ", "
                 << splitOutputShape[2] << ", " << splitOutputShape[3]
                 << "]\n");
    }

    // Create concat to combine outputs along channel dimension (axis=3 for
    // NHWC) Create signed i64 type for ONNX attributes (si64)
    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);
    auto concatOp = rewriter.create<ONNXConcatOp>(
        loc, outputType, concatInputs, IntegerAttr::get(si64Type, 3));

    rewriter.replaceOp(convOp, concatOp.getResult());

    LLVM_DEBUG(
        llvm::dbgs()
        << "Successfully split XFE depthwise conv with channel_multiplier="
        << channelMultiplier << " into " << channelMultiplier
        << " convs + concat\n");

    return success();
  }
};

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass to split depthwise conv2d operations with channel multiplier > 1
/// into multiple depthwise convs with channel_multiplier=1, followed by concat.
struct TransferDepthwiseConv2dWithChannelMultiplierPass
    : public PassWrapper<TransferDepthwiseConv2dWithChannelMultiplierPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transfer-depthwise-conv2d-with-channel-multiplier";
  }
  StringRef getDescription() const override {
    return "Split depthwise conv2d with channel_multiplier > 1 into multiple "
           "depthwise convs with channel_multiplier=1 followed by concat";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    // Add patterns for ONNX Conv (NCHW layout)
    patterns.add<SplitDepthwiseConvPattern>(ctx);

    // Add patterns for XFE Conv (NHWC layout)
    patterns.add<SplitXFEDepthwiseConvPattern>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass>
createTransferDepthwiseConv2dWithChannelMultiplierPass() {
  return std::make_unique<TransferDepthwiseConv2dWithChannelMultiplierPass>();
}

} // namespace onnx_mlir
