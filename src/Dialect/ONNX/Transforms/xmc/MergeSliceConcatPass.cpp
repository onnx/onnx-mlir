// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "onnx-merge-slice-concat"

using namespace mlir;

namespace {

// Templated helper to reorder 1D parameter data (e.g., InstanceNorm scale/bias)
template <typename T>
Value reorderParameterData(PatternRewriter &rewriter, Location loc,
    DenseElementsAttr denseAttr, RankedTensorType paramType,
    Type actualDataType, const SmallVector<int64_t> &channelReorderMap) {
  SmallVector<T> originalData(
      denseAttr.getValues<T>().begin(), denseAttr.getValues<T>().end());

  SmallVector<T> reorderedData;
  for (int64_t newIdx : channelReorderMap) {
    reorderedData.push_back(originalData[newIdx]);
  }

  // Create attribute with the storage type
  auto newAttrType =
      RankedTensorType::get(paramType.getShape(), actualDataType);
  auto newAttr =
      DenseElementsAttr::get(newAttrType, llvm::ArrayRef(reorderedData));

  // Create constant with the original type (which may include quant info)
  auto newConstOp = rewriter.create<ONNXConstantOp>(loc, paramType,
      /*sparse_value=*/Attribute(),
      /*value=*/newAttr, /*value_float=*/FloatAttr(),
      /*value_floats=*/ArrayAttr(), /*value_int=*/IntegerAttr(),
      /*value_ints=*/ArrayAttr(), /*value_string=*/StringAttr(),
      /*value_strings=*/ArrayAttr());
  return newConstOp.getResult();
}

// Templated helper to reorder conv weight data (4D tensor)
template <typename T>
SmallVector<T> reorderConvWeightData(const SmallVector<T> &originalData,
    const SmallVector<int64_t> &channelReorderMap, int64_t outChannels,
    int64_t inChannels, int64_t kernelH, int64_t kernelW, bool isNCHW) {
  SmallVector<T> reorderedData;
  reorderedData.reserve(originalData.size());

  if (isNCHW) {
    // For OIHW layout: iterate through out_ch, reorder in_ch, then kh, kw
    int64_t hw = kernelH * kernelW;
    for (int64_t oc = 0; oc < outChannels; ++oc) {
      for (int64_t ic = 0; ic < inChannels; ++ic) {
        int64_t oldIc = channelReorderMap[ic];
        int64_t srcBase = (oc * inChannels + oldIc) * hw;
        for (int64_t khw = 0; khw < hw; ++khw) {
          reorderedData.push_back(originalData[srcBase + khw]);
        }
      }
    }
  } else { // NHWC
    // For OHWI layout: iterate through out_ch, kh, kw, then reorder in_ch
    int64_t ohw = outChannels * kernelH * kernelW;
    for (int64_t i = 0; i < ohw; ++i) {
      int64_t baseIdx = i * inChannels;
      for (int64_t ic = 0; ic < inChannels; ++ic) {
        int64_t oldIc = channelReorderMap[ic];
        reorderedData.push_back(originalData[baseIdx + oldIc]);
      }
    }
  }

  return reorderedData;
}

// Helper function to check if a value is a constant with specific integer
// values
std::optional<SmallVector<int64_t>> getConstantIntegers(Value value) {
  auto constOp = value.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return std::nullopt;

  auto valueAttr = constOp.getValueAttr();
  if (!valueAttr)
    return std::nullopt;

  auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(valueAttr);
  if (!denseAttr)
    return std::nullopt;

  SmallVector<int64_t> result;
  for (auto val : denseAttr.getValues<int64_t>()) {
    result.push_back(val);
  }
  return result;
}

// Structure to hold slice information
struct SliceInfo {
  ONNXSliceOp op;
  int64_t beginChannel;
  int64_t endChannel;
  int64_t numChannels;
};

// Pattern to merge Slice->Concat->InstanceNorm->Conv pattern
struct MergeSliceConcatInstanceNormConv : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const override {
    // Check if conv input comes from InstanceNorm
    auto instanceNormOp =
        convOp.getX().getDefiningOp<ONNXInstanceNormalizationOp>();
    if (!instanceNormOp)
      return rewriter.notifyMatchFailure(
          convOp, "conv input does not come from InstanceNormalization");

    // Check if InstanceNorm input comes from concat
    auto concatOp = instanceNormOp.getInput().getDefiningOp<ONNXConcatOp>();
    if (!concatOp)
      return rewriter.notifyMatchFailure(
          convOp, "InstanceNormalization input does not come from Concat");

    auto axisAttr = concatOp.getAxisAttr();
    if (!axisAttr)
      return rewriter.notifyMatchFailure(
          convOp, "concat axis attribute missing");
    int64_t concatAxis = axisAttr.getValue().getSExtValue();

    // Handle channel concatenation (axis 1 for NCHW, axis 3 for NHWC)
    bool isNCHW = (concatAxis == 1);
    bool isNHWC = (concatAxis == 3);
    if (!isNCHW && !isNHWC)
      return rewriter.notifyMatchFailure(
          convOp, "concat axis is not channel axis (expected 1 for NCHW or 3 "
                  "for NHWC)");

    auto concatInputs = concatOp.getInputs();
    if (concatInputs.size() < 2)
      return rewriter.notifyMatchFailure(
          convOp, "concat has less than 2 inputs");

    SmallVector<SliceInfo> sliceInfos;
    Value commonInput;

    for (auto input : concatInputs) {
      auto sliceOp = input.getDefiningOp<ONNXSliceOp>();
      if (!sliceOp)
        return rewriter.notifyMatchFailure(
            convOp, "concat input does not come from Slice operation");

      if (!commonInput) {
        commonInput = sliceOp.getData();
      } else if (commonInput != sliceOp.getData()) {
        return rewriter.notifyMatchFailure(
            convOp, "slice operations do not share the same input");
      }

      auto starts = getConstantIntegers(sliceOp.getStarts());
      auto ends = getConstantIntegers(sliceOp.getEnds());
      auto axes = getConstantIntegers(sliceOp.getAxes());

      if (!starts || !ends || !axes)
        return rewriter.notifyMatchFailure(convOp,
            "slice parameters (starts/ends/axes) are not constant integers");

      int64_t channelAxisIdx = -1;
      for (size_t i = 0; i < axes->size(); ++i) {
        if ((*axes)[i] == concatAxis) {
          channelAxisIdx = i;
          break;
        }
      }

      if (channelAxisIdx < 0)
        return rewriter.notifyMatchFailure(
            convOp, "slice does not operate on the channel axis");

      int64_t beginChannel = (*starts)[channelAxisIdx];
      int64_t endChannel = (*ends)[channelAxisIdx];
      int64_t numChannels = endChannel - beginChannel;

      sliceInfos.push_back({sliceOp, beginChannel, endChannel, numChannels});
    }

    auto commonInputType =
        mlir::dyn_cast<RankedTensorType>(commonInput.getType());
    if (!commonInputType || !commonInputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          convOp, "common input is not a ranked tensor with static shape");

    auto concatOutputType =
        mlir::dyn_cast<RankedTensorType>(concatOp.getType());
    if (!concatOutputType || !concatOutputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          convOp, "concat output is not a ranked tensor with static shape");

    if (commonInputType.getShape() != concatOutputType.getShape())
      return rewriter.notifyMatchFailure(
          convOp, "common input shape does not match concat output shape");

    // Compute channel reordering
    SmallVector<std::pair<int64_t, size_t>> sortedIndices;
    for (size_t i = 0; i < sliceInfos.size(); ++i) {
      sortedIndices.push_back({sliceInfos[i].beginChannel, i});
    }
    llvm::sort(
        sortedIndices, [](auto &a, auto &b) { return a.first < b.first; });

    SmallVector<int64_t> channelReorderMap;
    for (auto [beginCh, sliceIdx] : sortedIndices) {
      int64_t numCh = sliceInfos[sliceIdx].numChannels;
      for (int64_t i = 0; i < numCh; ++i) {
        channelReorderMap.push_back(sliceInfos[sliceIdx].beginChannel + i);
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "MergeSliceConcatInstanceNormConv: Reordering "
                               "InstanceNorm params\n");

    // Reorder InstanceNorm scale and bias
    auto reorderInstanceNormParam = [&](Value param) -> Value {
      auto constOp = param.getDefiningOp<ONNXConstantOp>();
      if (!constOp)
        return Value();

      auto valueAttr = constOp.getValueAttr();
      if (!valueAttr)
        return Value();

      auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(valueAttr);
      if (!denseAttr)
        return Value();

      auto paramType = mlir::cast<RankedTensorType>(constOp.getType());
      auto actualDataType = denseAttr.getElementType();

      // Use templated helper function for type-specific reordering
      if (actualDataType.isF32()) {
        return reorderParameterData<float>(rewriter, constOp.getLoc(),
            denseAttr, paramType, actualDataType, channelReorderMap);
      } else if (actualDataType.isUnsignedInteger(8)) {
        return reorderParameterData<uint8_t>(rewriter, constOp.getLoc(),
            denseAttr, paramType, actualDataType, channelReorderMap);
      } else if (actualDataType.isSignedInteger(8)) {
        return reorderParameterData<int8_t>(rewriter, constOp.getLoc(),
            denseAttr, paramType, actualDataType, channelReorderMap);
      } else if (actualDataType.isUnsignedInteger(16)) {
        return reorderParameterData<uint16_t>(rewriter, constOp.getLoc(),
            denseAttr, paramType, actualDataType, channelReorderMap);
      } else if (actualDataType.isSignedInteger(16)) {
        return reorderParameterData<int16_t>(rewriter, constOp.getLoc(),
            denseAttr, paramType, actualDataType, channelReorderMap);
      }
      return Value();
    };

    auto newScale = reorderInstanceNormParam(instanceNormOp.getScale());
    auto newBias = reorderInstanceNormParam(instanceNormOp.getB());

    if (!newScale || !newBias)
      return rewriter.notifyMatchFailure(
          convOp, "failed to reorder InstanceNormalization parameters");

    // Create new InstanceNorm with original input
    auto newInstanceNorm = rewriter.create<ONNXInstanceNormalizationOp>(
        instanceNormOp.getLoc(), instanceNormOp.getType(), commonInput,
        newScale, newBias, instanceNormOp.getEpsilonAttr());

    // Now reorder conv weights
    auto weightsOp = convOp.getW().getDefiningOp<ONNXConstantOp>();
    if (!weightsOp || !weightsOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          convOp, "conv weights are not from a single-use constant operation");

    auto weightsAttr = weightsOp.getValueAttr();
    if (!weightsAttr)
      return rewriter.notifyMatchFailure(
          convOp, "conv weights constant has no value attribute");

    auto weightsData = mlir::dyn_cast<DenseElementsAttr>(weightsAttr);
    if (!weightsData)
      return rewriter.notifyMatchFailure(
          convOp, "conv weights are not dense elements");

    auto weightsType = mlir::cast<RankedTensorType>(weightsOp.getType());
    if (!weightsType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          convOp, "conv weights do not have static shape");

    // Conv weights shape:
    // NCHW: [out_channels, in_channels, kernel_h, kernel_w] (OIHW format)
    // NHWC: [out_channels, kernel_h, kernel_w, in_channels] (OHWI format)
    auto weightsShape = weightsType.getShape();
    if (weightsShape.size() != 4)
      return rewriter.notifyMatchFailure(
          convOp, "conv weights are not 4D (expected OIHW or OHWI format)");

    int64_t outChannels;
    int64_t inChannels;
    int64_t kernelH;
    int64_t kernelW;
    if (isNCHW) {
      outChannels = weightsShape[0];
      inChannels = weightsShape[1];
      kernelH = weightsShape[2];
      kernelW = weightsShape[3];
    } else { // NHWC
      outChannels = weightsShape[0];
      kernelH = weightsShape[1];
      kernelW = weightsShape[2];
      inChannels = weightsShape[3];
    }

    if (inChannels != static_cast<int64_t>(channelReorderMap.size()))
      return rewriter.notifyMatchFailure(
          convOp, "conv input channels do not match channel reorder map size");

    // Get the actual data type from the DenseElementsAttr
    auto actualDataType = weightsData.getElementType();

    // Helper lambda to create new conv after weight reordering
    auto createNewConv = [&](const auto &reorderedData) -> LogicalResult {
      // Create attribute with the storage type
      auto newWeightsAttrType =
          RankedTensorType::get(weightsType.getShape(), actualDataType);
      auto newWeightsAttr = DenseElementsAttr::get(
          newWeightsAttrType, llvm::ArrayRef(reorderedData));

      // Create constant with the original type (which may include quant info)
      auto newWeightsOp = rewriter.create<ONNXConstantOp>(weightsOp.getLoc(),
          weightsType, /*sparse_value=*/Attribute(),
          /*value=*/newWeightsAttr, /*value_float=*/FloatAttr(),
          /*value_floats=*/ArrayAttr(), /*value_int=*/IntegerAttr(),
          /*value_ints=*/ArrayAttr(), /*value_string=*/StringAttr(),
          /*value_strings=*/ArrayAttr());

      auto newConv =
          rewriter.create<ONNXConvOp>(convOp.getLoc(), convOp.getType(),
              newInstanceNorm.getResult(), newWeightsOp.getResult(),
              convOp.getB(), convOp.getAutoPadAttr(), convOp.getDilationsAttr(),
              convOp.getGroupAttr(), convOp.getKernelShapeAttr(),
              convOp.getPadsAttr(), convOp.getStridesAttr());

      rewriter.replaceOp(convOp, newConv.getResult());

      // Cleanup
      if (weightsOp->use_empty())
        rewriter.eraseOp(weightsOp);
      if (instanceNormOp->use_empty())
        rewriter.eraseOp(instanceNormOp);
      if (concatOp->use_empty())
        rewriter.eraseOp(concatOp);
      for (auto &info : sliceInfos) {
        if (info.op->use_empty())
          rewriter.eraseOp(info.op);
      }

      return success();
    };

    // Use templated helper function for type-specific weight reordering
    if (actualDataType.isF32()) {
      SmallVector<float> originalData(weightsData.getValues<float>().begin(),
          weightsData.getValues<float>().end());
      auto reorderedData = reorderConvWeightData<float>(originalData,
          channelReorderMap, outChannels, inChannels, kernelH, kernelW, isNCHW);
      return createNewConv(reorderedData);
    } else if (actualDataType.isUnsignedInteger(8)) {
      SmallVector<uint8_t> originalData(
          weightsData.getValues<uint8_t>().begin(),
          weightsData.getValues<uint8_t>().end());
      auto reorderedData = reorderConvWeightData<uint8_t>(originalData,
          channelReorderMap, outChannels, inChannels, kernelH, kernelW, isNCHW);
      return createNewConv(reorderedData);
    } else if (actualDataType.isSignedInteger(8)) {
      SmallVector<int8_t> originalData(weightsData.getValues<int8_t>().begin(),
          weightsData.getValues<int8_t>().end());
      auto reorderedData = reorderConvWeightData<int8_t>(originalData,
          channelReorderMap, outChannels, inChannels, kernelH, kernelW, isNCHW);
      return createNewConv(reorderedData);
    } else if (actualDataType.isUnsignedInteger(16)) {
      SmallVector<uint16_t> originalData(
          weightsData.getValues<uint16_t>().begin(),
          weightsData.getValues<uint16_t>().end());
      auto reorderedData = reorderConvWeightData<uint16_t>(originalData,
          channelReorderMap, outChannels, inChannels, kernelH, kernelW, isNCHW);
      return createNewConv(reorderedData);
    } else if (actualDataType.isSignedInteger(16)) {
      SmallVector<int16_t> originalData(
          weightsData.getValues<int16_t>().begin(),
          weightsData.getValues<int16_t>().end());
      auto reorderedData = reorderConvWeightData<int16_t>(originalData,
          channelReorderMap, outChannels, inChannels, kernelH, kernelW, isNCHW);
      return createNewConv(reorderedData);
    }

    return rewriter.notifyMatchFailure(convOp,
        "unsupported weight data type (expected f32, u8, i8, u16, or i16)");
  }
};

} // namespace

namespace onnx_mlir {

struct MergeSliceConcatPass
    : public PassWrapper<MergeSliceConcatPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "merge-slice-concat"; }
  StringRef getDescription() const override {
    return "Merge Slice-Concat patterns with downstream ops like InstanceNorm "
           "and Conv";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<MergeSliceConcatInstanceNormConv>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createMergeSliceConcatPass() {
  return std::make_unique<MergeSliceConcatPass>();
}

} // namespace onnx_mlir
