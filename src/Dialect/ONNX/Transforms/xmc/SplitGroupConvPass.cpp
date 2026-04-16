// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Slice per-axis quantization scales/zeroPoints for a sub-range of the
/// quantized dimension (e.g. when splitting weights along output channels).
/// Returns the original type unchanged if not per-axis quantized.
static Type slicePerAxisQuantType(
    Type elementType, int64_t start, int64_t count) {
  auto perAxisType = dyn_cast<quant::UniformQuantizedPerAxisType>(elementType);
  if (!perAxisType)
    return elementType;

  auto scales = perAxisType.getScales();
  auto zeroPoints = perAxisType.getZeroPoints();

  SmallVector<double> slicedScales(
      scales.begin() + start, scales.begin() + start + count);
  SmallVector<int64_t> slicedZeroPoints(
      zeroPoints.begin() + start, zeroPoints.begin() + start + count);

  return quant::UniformQuantizedPerAxisType::get(perAxisType.getFlags(),
      perAxisType.getStorageType(), perAxisType.getExpressedType(),
      slicedScales, slicedZeroPoints, perAxisType.getQuantizedDimension(),
      perAxisType.getStorageTypeMin(), perAxisType.getStorageTypeMax());
}

/// Create a constant tensor for integer values
Value createI64Constant(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> values) {
  auto tensorType = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, rewriter.getI64Type());
  auto attr = DenseElementsAttr::get(tensorType, values);
  return rewriter.create<ONNXConstantOp>(loc, tensorType,
      /*sparse_value=*/Attribute(), /*value=*/attr,
      /*value_float=*/FloatAttr(), /*value_floats=*/ArrayAttr(),
      /*value_int=*/IntegerAttr(), /*value_ints=*/ArrayAttr(),
      /*value_string=*/StringAttr(), /*value_strings=*/ArrayAttr());
}

/// Extract raw data from a constant op
template <typename T>
std::optional<SmallVector<T>> getConstantData(
    Value value, MLIRContext * /*context*/ = nullptr) {
  auto constOp = value.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return std::nullopt;

  auto valueAttr = constOp.getValueAttr();

  // Handle DenseElementsAttr
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
    SmallVector<T> data;
    for (auto val : denseAttr.getValues<T>()) {
      data.push_back(val);
    }
    return data;
  }

  // Handle DenseResourceElementsAttr - extract from raw buffer
  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(valueAttr)) {
    auto *blob = resourceAttr.getRawHandle().getBlob();
    if (!blob) {
      // Blob not loaded - cannot extract data
      // Note: The blob should be loaded when the MLIR file is parsed.
      // If it's not available, we cannot proceed with the transformation.
      return std::nullopt;
    }

    // Get raw data buffer
    ArrayRef<char> rawData = blob->getData();
    if (rawData.empty())
      return std::nullopt;

    // Materialize as DenseElementsAttr first to handle endianness and type
    // conversion
    auto denseAttr =
        DenseElementsAttr::getFromRawBuffer(resourceAttr.getType(), rawData);
    if (!denseAttr)
      return std::nullopt;

    // Extract from the materialized dense attribute
    SmallVector<T> data;
    for (auto val : denseAttr.getValues<T>()) {
      data.push_back(val);
    }

    return data;
  }

  return std::nullopt;
}

/// Extract storage type from potentially quantized type
Type getStorageType(Type type) {
  if (auto quantType = dyn_cast<quant::QuantizedType>(type)) {
    return quantType.getStorageType();
  }
  return type;
}

/// Get policy for splitting group convolution
/// Returns the number of splits to create
int64_t getGroupConvSplitPolicy(ONNXConvOp convOp) {
  // Get input and weight shapes
  auto inputType = dyn_cast<RankedTensorType>(convOp.getX().getType());
  auto weightType = dyn_cast<RankedTensorType>(convOp.getW().getType());

  if (!inputType || !weightType)
    return 1;

  auto inputShape = inputType.getShape();
  auto weightShape = weightType.getShape();

  // ONNX uses NCHW format, so channel is dimension 1
  if (inputShape.size() < 2 || weightShape.size() < 2)
    return 1;

  int64_t inputChannels = inputShape[1];
  int64_t weightChannels = weightShape[1]; // channels per group

  // Get current group count
  int64_t currentGroup = convOp.getGroup();

  // Formula: input_channels / (2 * weight_channels)
  int64_t splitCount = inputChannels / (2 * weightChannels);

  // Validate: must be at least 1, at most currentGroup, and divide evenly
  if (splitCount < 1 || splitCount > currentGroup ||
      currentGroup % splitCount != 0) {
    return 1; // No split
  }

  return splitCount;
}

//===----------------------------------------------------------------------===//
// SplitGroupConvPattern
//===----------------------------------------------------------------------===//

struct SplitGroupConvPattern : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const override {
    auto loc = convOp.getLoc();

    // 1. Check if this is a group convolution (group > 1)
    int64_t currentGroup = convOp.getGroup();
    if (currentGroup <= 1) {
      return failure(); // Not a group conv
    }

    // 2. Get split policy
    int64_t splitCount = getGroupConvSplitPolicy(convOp);
    if (splitCount <= 1) {
      return failure(); // No benefit to splitting
    }

    // 3. Get input, weight, and bias
    Value input = convOp.getX();
    Value weights = convOp.getW();
    Value bias = convOp.getB();

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto weightType = dyn_cast<RankedTensorType>(weights.getType());
    auto outputType = dyn_cast<RankedTensorType>(convOp.getType());

    if (!inputType || !weightType || !outputType) {
      return failure();
    }

    auto inputShape = inputType.getShape();
    auto weightShape = weightType.getShape();
    auto outputShape = outputType.getShape();

    // NCHW format
    int64_t batch = inputShape[0];
    int64_t inputChannels = inputShape[1];
    int64_t height = inputShape[2];
    int64_t width = inputShape[3];

    int64_t outputChannels = outputShape[1];
    int64_t outHeight = outputShape[2];
    int64_t outWidth = outputShape[3];

    // Channels per split
    int64_t inputChannelsPerSplit = inputChannels / splitCount;
    int64_t outputChannelsPerSplit = outputChannels / splitCount;
    int64_t newGroupSize = currentGroup / splitCount;

    // 4. Get weight constant data
    auto weightsConstOp = weights.getDefiningOp<ONNXConstantOp>();
    if (!weightsConstOp) {
      return failure();
    }

    // Get element type (should be quantized)
    auto weightElemType = weightType.getElementType();

    // Convert dense_resource to dense if needed, then extract data
    auto valueAttr = weightsConstOp.getValueAttr();
    DenseElementsAttr weightDenseAttr;

    if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
      weightDenseAttr = denseAttr;
    } else if (auto resourceAttr =
                   dyn_cast<DenseResourceElementsAttr>(valueAttr)) {
      auto handle = resourceAttr.getRawHandle();
      auto *blob = handle.getBlob();
      if (!blob) {
        // Blob not loaded - cannot proceed
        // The blob should be loaded when MLIR file is parsed
        // If it's not available, we cannot extract the data
        return failure();
      }
      // Convert dense_resource to dense elements
      ArrayRef<char> rawData = blob->getData();
      if (rawData.empty()) {
        return failure();
      }
      weightDenseAttr =
          DenseElementsAttr::getFromRawBuffer(resourceAttr.getType(), rawData);
      if (!weightDenseAttr) {
        return failure();
      }
      // Replace the constant with dense version
      auto newWeightConst = rewriter.create<ONNXConstantOp>(loc, weightType,
          /*sparse_value=*/Attribute(),
          /*value=*/weightDenseAttr, /*value_float=*/FloatAttr(),
          /*value_floats=*/ArrayAttr(), /*value_int=*/IntegerAttr(),
          /*value_ints=*/ArrayAttr(), /*value_string=*/StringAttr(),
          /*value_strings=*/ArrayAttr());
      weights = newWeightConst.getResult();
      weightsConstOp = newWeightConst;
    } else {
      return failure();
    }

    // TODO: Handle weight data in different types
    //  Extract weight data as uint8
    SmallVector<uint8_t> weightData;
    for (auto val : weightDenseAttr.getValues<uint8_t>()) {
      weightData.push_back(val);
    }

    // 5. Get bias constant data (if exists)
    // TODO: Handle bias data in different types
    SmallVector<uint32_t> biasData;
    bool hasBias = false;
    Type biasElemType;
    if (bias) {
      auto biasType = dyn_cast<RankedTensorType>(bias.getType());
      if (biasType) {
        auto biasConstOp = bias.getDefiningOp<ONNXConstantOp>();
        if (biasConstOp) {
          // Convert dense_resource to dense if needed
          auto biasValueAttr = biasConstOp.getValueAttr();
          DenseElementsAttr biasDenseAttr;

          if (auto denseAttr = dyn_cast<DenseElementsAttr>(biasValueAttr)) {
            biasDenseAttr = denseAttr;
          } else if (auto resourceAttr =
                         dyn_cast<DenseResourceElementsAttr>(biasValueAttr)) {
            auto handle = resourceAttr.getRawHandle();
            auto *blob = handle.getBlob();
            if (blob) {
              biasDenseAttr = DenseElementsAttr::getFromRawBuffer(
                  resourceAttr.getType(), blob->getData());
              if (biasDenseAttr) {
                // Replace the constant with dense version
                auto newBiasConst = rewriter.create<ONNXConstantOp>(loc,
                    biasType, /*sparse_value=*/Attribute(),
                    /*value=*/biasDenseAttr, /*value_float=*/FloatAttr(),
                    /*value_floats=*/ArrayAttr(), /*value_int=*/IntegerAttr(),
                    /*value_ints=*/ArrayAttr(), /*value_string=*/StringAttr(),
                    /*value_strings=*/ArrayAttr());
                bias = newBiasConst.getResult();
                biasConstOp = newBiasConst;
              }
            }
          }

          if (biasDenseAttr) {
            for (auto val : biasDenseAttr.getValues<uint32_t>()) {
              biasData.push_back(val);
            }
            hasBias = true;
            biasElemType = biasType.getElementType();
          }
        }
      }
    }

    // 6. Create slice operations and new convolutions for each split
    SmallVector<Value> convResults;

    // Weight dimensions: [CO, CI_per_group, KH, KW] in NCHW
    int64_t CI_per_group = weightShape[1];
    int64_t KH = weightShape[2];
    int64_t KW = weightShape[3];

    for (int64_t i = 0; i < splitCount; ++i) {
      // Calculate channel ranges for this split
      int64_t inputStartChannel = i * inputChannelsPerSplit;
      int64_t inputEndChannel = (i + 1) * inputChannelsPerSplit;
      int64_t outputStartChannel = i * outputChannelsPerSplit;

      // Create slice operation for input channels
      Value starts =
          createI64Constant(rewriter, loc, {0, inputStartChannel, 0, 0});
      Value ends = createI64Constant(
          rewriter, loc, {batch, inputEndChannel, height, width});
      Value axes = createI64Constant(rewriter, loc, {0, 1, 2, 3});
      Value steps = createI64Constant(rewriter, loc, {1, 1, 1, 1});

      auto slicedInputType =
          RankedTensorType::get({batch, inputChannelsPerSplit, height, width},
              inputType.getElementType());

      auto slicedInput = rewriter.create<ONNXSliceOp>(
          loc, slicedInputType, input, starts, ends, axes, steps);

      // Create new weight constant for this split
      // Extract the relevant portion of weights
      int64_t newWeightCO = outputChannelsPerSplit;
      int64_t newWeightCI = inputChannelsPerSplit / newGroupSize;

      SmallVector<uint8_t> newWeightData(newWeightCO * newWeightCI * KH * KW);

      // Copy weight data for this split
      // Original: CO x CI_per_group x KH x KW
      // New: newWeightCO x newWeightCI x KH x KW
      for (int64_t co = 0; co < newWeightCO; ++co) {
        int64_t src_co = outputStartChannel + co;
        for (int64_t kh = 0; kh < KH; ++kh) {
          for (int64_t kw = 0; kw < KW; ++kw) {
            for (int64_t ci = 0; ci < newWeightCI; ++ci) {
              // Source offset in original weights
              int64_t src_offset = src_co * CI_per_group * KH * KW +
                                   kh * KW * CI_per_group + kw * CI_per_group +
                                   ci;

              // Destination offset in new weights
              int64_t dst_offset = co * newWeightCI * KH * KW +
                                   kh * KW * newWeightCI + kw * newWeightCI +
                                   ci;

              if (src_offset < static_cast<int64_t>(weightData.size()) &&
                  dst_offset < static_cast<int64_t>(newWeightData.size())) {
                newWeightData[dst_offset] = weightData[src_offset];
              }
            }
          }
        }
      }

      // Create new weight constant
      auto newWeightShape =
          SmallVector<int64_t>{newWeightCO, newWeightCI, KH, KW};

      // Get storage type for creating DenseElementsAttr
      auto weightStorageType = getStorageType(weightElemType);
      auto weightStorageTensorType =
          RankedTensorType::get(newWeightShape, weightStorageType);
      auto newWeightAttr = DenseElementsAttr::get(
          weightStorageTensorType, ArrayRef<uint8_t>(newWeightData));

      // Slice per-axis quant scales/zp for this split's output channels
      auto splitWeightElemType = slicePerAxisQuantType(
          weightElemType, outputStartChannel, newWeightCO);
      auto newWeightType =
          RankedTensorType::get(newWeightShape, splitWeightElemType);

      auto newWeightConst = rewriter.create<ONNXConstantOp>(loc, newWeightType,
          /*sparse_value=*/Attribute(),
          /*value=*/newWeightAttr, /*value_float=*/FloatAttr(),
          /*value_floats=*/ArrayAttr(), /*value_int=*/IntegerAttr(),
          /*value_ints=*/ArrayAttr(), /*value_string=*/StringAttr(),
          /*value_strings=*/ArrayAttr());

      // Create new bias constant for this split (if applicable)
      Value newBias;
      if (hasBias) {
        SmallVector<uint32_t> newBiasData(outputChannelsPerSplit);
        for (int64_t j = 0; j < outputChannelsPerSplit; ++j) {
          int64_t src_idx = outputStartChannel + j;
          if (src_idx < static_cast<int64_t>(biasData.size())) {
            newBiasData[j] = biasData[src_idx];
          }
        }

        // Get storage type for creating DenseElementsAttr
        auto biasStorageType = getStorageType(biasElemType);
        auto biasStorageTensorType =
            RankedTensorType::get({outputChannelsPerSplit}, biasStorageType);
        auto newBiasAttr = DenseElementsAttr::get(
            biasStorageTensorType, ArrayRef<uint32_t>(newBiasData));

        // Slice per-axis quant scales/zp for this split's bias channels
        auto splitBiasElemType = slicePerAxisQuantType(
            biasElemType, outputStartChannel, outputChannelsPerSplit);
        auto newBiasType =
            RankedTensorType::get({outputChannelsPerSplit}, splitBiasElemType);

        newBias = rewriter.create<ONNXConstantOp>(loc, newBiasType,
            /*sparse_value=*/Attribute(),
            /*value=*/newBiasAttr, /*value_float=*/FloatAttr(),
            /*value_floats=*/ArrayAttr(), /*value_int=*/IntegerAttr(),
            /*value_ints=*/ArrayAttr(), /*value_string=*/StringAttr(),
            /*value_strings=*/ArrayAttr());
      }

      // Create new conv operation
      auto newConvOutputType = RankedTensorType::get(
          {batch, outputChannelsPerSplit, outHeight, outWidth},
          outputType.getElementType());

      // Create group attribute with same type as original
      auto groupAttr = IntegerAttr::get(
          IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed),
          newGroupSize);

      auto newConvOp = rewriter.create<ONNXConvOp>(loc, newConvOutputType,
          slicedInput.getResult(), newWeightConst.getResult(), newBias,
          convOp.getAutoPadAttr(), convOp.getDilationsAttr(), groupAttr,
          convOp.getKernelShapeAttr(), convOp.getPadsAttr(),
          convOp.getStridesAttr());

      convResults.push_back(newConvOp.getResult());
    }

    // 7. Create concat operation to merge outputs
    auto axisAttr = IntegerAttr::get(
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed), 1);
    auto concatOp =
        rewriter.create<ONNXConcatOp>(loc, outputType, convResults, axisAttr);

    // 8. Replace original conv with concat result
    rewriter.replaceOp(convOp, concatOp.getResult());

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct SplitGroupConvPass
    : public PassWrapper<SplitGroupConvPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "split-group-conv"; }
  StringRef getDescription() const override {
    return "Split group convolutions into multiple smaller convolutions";
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Ensure dense_resource blobs are loaded by accessing the module
    // The blobs should be loaded when the MLIR file is parsed, but we need
    // to ensure they're available in the context
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    if (moduleOp) {
      // Access the module to ensure blobs are loaded
      (void)moduleOp;
    }

    RewritePatternSet patterns(context);
    patterns.add<SplitGroupConvPattern>(context);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createSplitGroupConvPass() {
  return std::make_unique<SplitGroupConvPass>();
}

} // namespace onnx_mlir
