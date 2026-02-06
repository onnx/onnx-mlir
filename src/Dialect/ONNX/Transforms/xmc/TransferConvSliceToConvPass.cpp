//===----------------------------------------------------------------------===//
// TransferConvSliceToConvPass
//
// This pass transforms Conv -> Slice patterns by moving the slice operation
// before the convolution, adjusting weights, bias, and padding as needed.
//
// PREREQUISITE: StandardizeSliceOpsPass must run before this pass.
// This pass assumes slices are in standardized format:
// - starts, ends, steps are dense tensors covering all dimensions
// - axes is [0, 1, 2, ..., rank-1]
// - No negative indices (already resolved)
//===----------------------------------------------------------------------===//

// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

#include "llvm/ADT/SmallVector.h"

#include <cmath>
#include <iostream>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get shape from a value
static llvm::SmallVector<int64_t> getShape(mlir::Value value) {
  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasRank())
    return {};
  return llvm::SmallVector<int64_t>(
      shapedType.getShape().begin(), shapedType.getShape().end());
}

/// Get element type from a value
static mlir::Type getElementType(mlir::Value value) {
  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(value.getType());
  if (!shapedType)
    return nullptr;
  return shapedType.getElementType();
}

/// Extract scale and zero point from a quantized type.
/// Returns true if successful, false if the type is not quantized.
static bool getQuantParams(mlir::Type type, double &scale, int64_t &zeroPoint) {
  if (auto quantType =
          mlir::dyn_cast<mlir::quant::UniformQuantizedType>(type)) {
    scale = quantType.getScale();
    zeroPoint = quantType.getZeroPoint();
    return true;
  }
  return false;
}

/// Extract scale and zero point from the element type of a value.
static bool getQuantParamsFromValue(
    mlir::Value value, double &scale, int64_t &zeroPoint) {
  auto elemType = getElementType(value);
  if (!elemType)
    return false;
  return getQuantParams(elemType, scale, zeroPoint);
}

/// Check if the slice input and output have matching quantization parameters.
/// Both input and output quant params are extracted from tensor type encoding.
/// Returns true if quant params match or if there's no quantization to check.
static bool sliceQuantParamsMatch(mlir::ONNXSliceOp sliceOp) {
  // Get quant params from slice input type (conv output type)
  double inputScale = 0.0;
  int64_t inputZp = 0;
  bool hasInputQuant =
      getQuantParamsFromValue(sliceOp.getData(), inputScale, inputZp);

  // Get quant params from slice output type
  double outputScale = 0.0;
  int64_t outputZp = 0;
  bool hasOutputQuant =
      getQuantParamsFromValue(sliceOp.getOutput(), outputScale, outputZp);

  // If neither input nor output is quantized, allow the transformation
  if (!hasInputQuant && !hasOutputQuant)
    return true;

  // If only one side is quantized, don't allow transformation
  // (mismatched quantization state)
  if (hasInputQuant != hasOutputQuant)
    return false;

  // Both are quantized - compare with tolerance for floating point
  constexpr double tolerance = 1e-6;
  bool scaleMatch = std::abs(inputScale - outputScale) < tolerance;
  bool zpMatch = (inputZp == outputZp);

  return scaleMatch && zpMatch;
}

/// Try to extract constant integer array from a Value
static bool getConstantIntArray(
    mlir::Value value, llvm::SmallVector<int64_t> &result) {
  auto defOp = value.getDefiningOp<mlir::ONNXConstantOp>();
  if (!defOp)
    return false;

  auto valueAttr = defOp.getValueAttr();
  if (!valueAttr)
    return false;

  auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr);
  if (!denseAttr)
    return false;

  result.clear();
  for (auto val : denseAttr.getValues<mlir::APInt>()) {
    result.push_back(val.getSExtValue());
  }
  return true;
}

/// Extract float data from constant op
static bool getConstantFloatData(
    mlir::ONNXConstantOp constOp, llvm::SmallVector<float> &result) {
  auto valueAttr = constOp.getValueAttr();
  if (!valueAttr)
    return false;

  auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr);
  if (!denseAttr)
    return false;

  result.clear();
  for (auto val : denseAttr.getValues<mlir::APFloat>()) {
    result.push_back(val.convertToFloat());
  }
  return true;
}

/// Extract integer data from constant op (for int8/uint8 quantized weights)
static bool getConstantIntData(
    mlir::ONNXConstantOp constOp, llvm::SmallVector<int64_t> &result) {
  auto valueAttr = constOp.getValueAttr();
  if (!valueAttr)
    return false;

  auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr);
  if (!denseAttr)
    return false;

  result.clear();
  for (auto val : denseAttr.getValues<mlir::APInt>()) {
    result.push_back(val.getSExtValue());
  }
  return true;
}

/// Check if a type is a quantized type
static bool isQuantizedType(mlir::Type type) {
  if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type)) {
    return mlir::isa<mlir::quant::QuantizedType>(shapedType.getElementType());
  }
  return mlir::isa<mlir::quant::QuantizedType>(type);
}

/// Slice integer weight data for channel selection
static llvm::SmallVector<int64_t> sliceIntWeightData(
    llvm::ArrayRef<int64_t> originalData, llvm::ArrayRef<int64_t> originalShape,
    int64_t beginChannel, int64_t endChannel, int64_t strideChannel) {
  int64_t elementsPerFilter = 1;
  for (size_t i = 1; i < originalShape.size(); ++i) {
    elementsPerFilter *= originalShape[i];
  }

  llvm::SmallVector<int64_t> result;
  if (strideChannel > 0) {
    for (int64_t ch = beginChannel; ch < endChannel; ch += strideChannel) {
      int64_t startIdx = ch * elementsPerFilter;
      for (int64_t i = 0; i < elementsPerFilter; ++i) {
        result.push_back(originalData[startIdx + i]);
      }
    }
  } else if (strideChannel < 0) {
    for (int64_t ch = beginChannel; ch > endChannel; ch += strideChannel) {
      int64_t startIdx = ch * elementsPerFilter;
      for (int64_t i = 0; i < elementsPerFilter; ++i) {
        result.push_back(originalData[startIdx + i]);
      }
    }
  }
  return result;
}

/// Create a constant tensor from integer data with quantized type
static mlir::Value createConstantFromIntData(mlir::PatternRewriter &rewriter,
    mlir::Location loc, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> data, mlir::Type elementType) {
  // Get the storage type from quantized type if needed
  mlir::Type storageType = elementType;
  if (auto quantType =
          mlir::dyn_cast<mlir::quant::QuantizedType>(elementType)) {
    storageType = quantType.getStorageType();
  }

  auto tensorType = mlir::RankedTensorType::get(shape, elementType);
  auto storageTensorType = mlir::RankedTensorType::get(shape, storageType);

  // Convert int64_t data to the appropriate storage type
  llvm::SmallVector<mlir::APInt> apIntData;
  unsigned bitWidth = storageType.getIntOrFloatBitWidth();
  for (int64_t val : data) {
    apIntData.push_back(mlir::APInt(bitWidth, val, /*isSigned=*/true));
  }

  auto denseAttr =
      mlir::DenseIntElementsAttr::get(storageTensorType, apIntData);
  return rewriter.create<mlir::ONNXConstantOp>(loc, tensorType,
      mlir::ValueRange{},
      mlir::ArrayRef<mlir::NamedAttribute>{
          rewriter.getNamedAttr("value", denseAttr)});
}

/// Create a constant tensor from float data
static mlir::Value createConstantFromFloatData(mlir::PatternRewriter &rewriter,
    mlir::Location loc, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<float> data, mlir::Type elementType) {
  auto tensorType = mlir::RankedTensorType::get(shape, elementType);
  auto denseAttr = mlir::DenseElementsAttr::get(tensorType, data);
  return rewriter.create<mlir::ONNXConstantOp>(loc, tensorType,
      mlir::ValueRange{},
      mlir::ArrayRef<mlir::NamedAttribute>{
          rewriter.getNamedAttr("value", denseAttr)});
}

/// Create a constant i64 tensor
static mlir::Value createConstantI64Array(mlir::PatternRewriter &rewriter,
    mlir::Location loc, llvm::ArrayRef<int64_t> values) {
  auto tensorType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, rewriter.getI64Type());
  auto denseAttr = mlir::DenseIntElementsAttr::get(tensorType, values);
  return rewriter.create<mlir::ONNXConstantOp>(loc, tensorType,
      mlir::ValueRange{},
      mlir::ArrayRef<mlir::NamedAttribute>{
          rewriter.getNamedAttr("value", denseAttr)});
}

/// Check if Conv is a standard (non-grouped) convolution
static bool isNonGroupedConv(mlir::ONNXConvOp convOp) {
  auto groupAttr = convOp.getGroupAttr();
  if (!groupAttr)
    return true; // Default is group=1
  return groupAttr.getValue().getSExtValue() == 1;
}

/// Slice weight tensor for channel selection
/// Weight shape: [output_channels, input_channels/group, kH, kW]
static llvm::SmallVector<float> sliceWeightData(
    llvm::ArrayRef<float> originalData, llvm::ArrayRef<int64_t> originalShape,
    int64_t beginChannel, int64_t endChannel, int64_t strideChannel) {

  int64_t elementsPerFilter = 1;
  for (size_t i = 1; i < originalShape.size(); ++i) {
    elementsPerFilter *= originalShape[i];
  }

  llvm::SmallVector<float> result;

  if (strideChannel > 0) {
    for (int64_t ch = beginChannel; ch < endChannel; ch += strideChannel) {
      int64_t startIdx = ch * elementsPerFilter;
      for (int64_t i = 0; i < elementsPerFilter; ++i) {
        result.push_back(originalData[startIdx + i]);
      }
    }
  } else if (strideChannel < 0) {
    for (int64_t ch = beginChannel; ch > endChannel; ch += strideChannel) {
      int64_t startIdx = ch * elementsPerFilter;
      for (int64_t i = 0; i < elementsPerFilter; ++i) {
        result.push_back(originalData[startIdx + i]);
      }
    }
  }

  return result;
}

/// Slice bias tensor for channel selection (float version)
/// Bias shape: [output_channels]
static llvm::SmallVector<float> sliceBiasData(
    llvm::ArrayRef<float> originalData, int64_t beginChannel,
    int64_t endChannel, int64_t strideChannel) {

  llvm::SmallVector<float> result;

  if (strideChannel > 0) {
    for (int64_t ch = beginChannel; ch < endChannel; ch += strideChannel) {
      result.push_back(originalData[ch]);
    }
  } else if (strideChannel < 0) {
    for (int64_t ch = beginChannel; ch > endChannel; ch += strideChannel) {
      result.push_back(originalData[ch]);
    }
  }

  return result;
}

/// Slice bias tensor for channel selection (integer version for quantized bias)
/// Bias shape: [output_channels]
static llvm::SmallVector<int64_t> sliceIntBiasData(
    llvm::ArrayRef<int64_t> originalData, int64_t beginChannel,
    int64_t endChannel, int64_t strideChannel) {

  llvm::SmallVector<int64_t> result;

  if (strideChannel > 0) {
    for (int64_t ch = beginChannel; ch < endChannel; ch += strideChannel) {
      result.push_back(originalData[ch]);
    }
  } else if (strideChannel < 0) {
    for (int64_t ch = beginChannel; ch > endChannel; ch += strideChannel) {
      result.push_back(originalData[ch]);
    }
  }

  return result;
}

/// Calculate number of output channels after slicing
static int64_t calculateSlicedChannels(
    int64_t begin, int64_t end, int64_t stride) {
  if (stride > 0) {
    return (end - begin + stride - 1) / stride;
  } else if (stride < 0) {
    return (begin - end - stride - 1) / (-stride);
  }
  return 0;
}

/// Check if slice affects spatial (H/W) dimensions
static bool hasSpatialSlice(llvm::ArrayRef<int64_t> inputShape,
    llvm::ArrayRef<int64_t> outputShape,
    [[maybe_unused]] llvm::ArrayRef<int64_t> steps) {
  // Check spatial dimensions (H, W are typically at indices 2, 3)
  for (size_t i = 2; i < inputShape.size(); ++i) {
    if (inputShape[i] != outputShape[i])
      return true;
  }
  return false;
}

/// Check if spatial strides are all 1 (no striding, only cropping)
static bool spatialStridesAreOne(llvm::ArrayRef<int64_t> steps) {
  // Steps for spatial dims (indices 2, 3, ...)
  for (size_t i = 2; i < steps.size(); ++i) {
    if (steps[i] != 1)
      return false;
  }
  return true;
}

/// Check if batch dimension is unchanged
static bool batchUnchanged(llvm::ArrayRef<int64_t> inputShape,
    llvm::ArrayRef<int64_t> outputShape, llvm::ArrayRef<int64_t> steps) {
  if (inputShape[0] != outputShape[0])
    return false;
  if (!steps.empty() && steps[0] != 1)
    return false;
  return true;
}

/// Get conv attributes as vectors
static bool getConvAttributes(mlir::ONNXConvOp convOp,
    llvm::SmallVector<int64_t> &strides, llvm::SmallVector<int64_t> &pads,
    llvm::SmallVector<int64_t> &dilations,
    llvm::SmallVector<int64_t> &kernelShape) {
  // Get strides (default [1, 1])
  if (auto stridesAttr = convOp.getStridesAttr()) {
    for (mlir::Attribute s : stridesAttr.getValue())
      strides.push_back(
          mlir::cast<mlir::IntegerAttr>(s).getValue().getSExtValue());
  } else {
    strides = {1, 1};
  }

  // Get pads (default [0, 0, 0, 0])
  if (auto padsAttr = convOp.getPadsAttr()) {
    for (mlir::Attribute p : padsAttr.getValue())
      pads.push_back(
          mlir::cast<mlir::IntegerAttr>(p).getValue().getSExtValue());
  } else {
    pads = {0, 0, 0, 0};
  }

  // Get dilations (default [1, 1])
  if (auto dilationsAttr = convOp.getDilationsAttr()) {
    for (mlir::Attribute d : dilationsAttr.getValue())
      dilations.push_back(
          mlir::cast<mlir::IntegerAttr>(d).getValue().getSExtValue());
  } else {
    dilations = {1, 1};
  }

  // Get kernel shape
  if (auto kernelAttr = convOp.getKernelShapeAttr()) {
    for (mlir::Attribute k : kernelAttr.getValue())
      kernelShape.push_back(
          mlir::cast<mlir::IntegerAttr>(k).getValue().getSExtValue());
  } else {
    // Try to infer from weights
    return false;
  }

  return true;
}

/// Calculate input region needed to produce a specific output region
/// Based on convolution formula: out_idx = (in_idx + pad - dilation * (kernel -
/// 1) - 1) / stride + 1 Inverse: in_idx = stride * out_idx - pad + dilation *
/// (kernel - 1)
struct SpatialSliceParams {
  int64_t inputBeginH, inputEndH;
  int64_t inputBeginW, inputEndW;
  int64_t newPadTop, newPadBottom;
  int64_t newPadLeft, newPadRight;
};

static SpatialSliceParams calculateSpatialSliceParams(
    llvm::ArrayRef<int64_t> convInputShape, // Conv's input shape [N, C, H, W]
    llvm::ArrayRef<int64_t> outputBegin,    // [beginH, beginW]
    llvm::ArrayRef<int64_t> outputEnd,      // [endH, endW]
    llvm::ArrayRef<int64_t> strides,        // [strideH, strideW]
    llvm::ArrayRef<int64_t> pads,      // [padTop, padLeft, padBottom, padRight]
    llvm::ArrayRef<int64_t> dilations, // [dilationH, dilationW]
    llvm::ArrayRef<int64_t> kernelShape) { // [kernelH, kernelW]

  SpatialSliceParams result;

  // ONNX pads format: [x1_begin, x2_begin, x1_end, x2_end] for 2D
  // For NCHW: [pad_top, pad_left, pad_bottom, pad_right] OR [pad_H_begin,
  // pad_W_begin, pad_H_end, pad_W_end] Convention: pads[0]=pad_H_begin,
  // pads[1]=pad_W_begin, pads[2]=pad_H_end, pads[3]=pad_W_end
  int64_t padTop = pads.size() > 0 ? pads[0] : 0;
  int64_t padLeft = pads.size() > 1 ? pads[1] : 0;

  int64_t strideH = strides[0];
  int64_t strideW = strides[1];
  int64_t dilationH = dilations[0];
  int64_t dilationW = dilations[1];
  int64_t kernelH = kernelShape[0];
  int64_t kernelW = kernelShape[1];

  int64_t inputH = convInputShape[2];
  int64_t inputW = convInputShape[3];

  // Calculate input range needed for the output slice
  // For output pixel at out_idx, we need input pixels from:
  //   in_start = stride * out_idx - pad
  //   in_end = stride * out_idx - pad + dilation * (kernel - 1) + 1

  // Height dimension
  int64_t beginH_raw = strideH * outputBegin[0] - padTop;
  int64_t endH_raw =
      strideH * (outputEnd[0] - 1) + dilationH * (kernelH - 1) - padTop + 1;

  // Handle case where kernel < stride (gaps between receptive fields)
  if (kernelH < strideH) {
    endH_raw += (strideH - kernelH);
  }

  // Width dimension
  int64_t beginW_raw = strideW * outputBegin[1] - padLeft;
  int64_t endW_raw =
      strideW * (outputEnd[1] - 1) + dilationW * (kernelW - 1) - padLeft + 1;

  if (kernelW < strideW) {
    endW_raw += (strideW - kernelW);
  }

  // Clamp to valid input range
  result.inputBeginH = std::max(beginH_raw, (int64_t)0);
  result.inputEndH = std::min(endH_raw, inputH);
  result.inputBeginW = std::max(beginW_raw, (int64_t)0);
  result.inputEndW = std::min(endW_raw, inputW);

  // Ensure valid range
  result.inputBeginH = std::min(result.inputBeginH, inputH);
  result.inputEndH = std::max(result.inputEndH, result.inputBeginH);
  result.inputBeginW = std::min(result.inputBeginW, inputW);
  result.inputEndW = std::max(result.inputEndW, result.inputBeginW);

  // Calculate new padding needed after slicing input
  // New padding compensates for any part of the receptive field that was
  // outside the original input
  result.newPadTop = std::max(-beginH_raw, (int64_t)0);
  result.newPadBottom = std::max(endH_raw - inputH, (int64_t)0);
  result.newPadLeft = std::max(-beginW_raw, (int64_t)0);
  result.newPadRight = std::max(endW_raw - inputW, (int64_t)0);

  return result;
}

/// Create a Slice operation
static mlir::Value createSliceOp(mlir::PatternRewriter &rewriter,
    mlir::Location loc, mlir::Value input, llvm::ArrayRef<int64_t> starts,
    llvm::ArrayRef<int64_t> ends, llvm::ArrayRef<int64_t> axes,
    llvm::ArrayRef<int64_t> steps,
    mlir::RankedTensorType referenceType = nullptr) {

  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto inputShape = getShape(input);

  // Calculate output shape
  llvm::SmallVector<int64_t> outputShape = inputShape;
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    int64_t start = starts[i];
    int64_t end = ends[i];
    int64_t step = steps[i];
    outputShape[axis] = (end - start + step - 1) / step;
  }

  // Slice preserves input's element type (slice doesn't change element type)
  // Only copy encoding (quantization metadata) from reference type if provided
  mlir::Type elemType = inputType.getElementType();
  mlir::Attribute encoding =
      referenceType ? referenceType.getEncoding() : inputType.getEncoding();

  auto outputType =
      mlir::RankedTensorType::get(outputShape, elemType, encoding);

  // Create constant tensors for slice parameters
  mlir::Value startsConst = createConstantI64Array(rewriter, loc, starts);
  mlir::Value endsConst = createConstantI64Array(rewriter, loc, ends);
  mlir::Value axesConst = createConstantI64Array(rewriter, loc, axes);
  mlir::Value stepsConst = createConstantI64Array(rewriter, loc, steps);

  return rewriter.create<mlir::ONNXSliceOp>(
      loc, outputType, input, startsConst, endsConst, axesConst, stepsConst);
}

//===----------------------------------------------------------------------===//
// TransferConvSliceToConvPattern
//===----------------------------------------------------------------------===//

struct TransferConvSliceToConvPattern
    : public mlir::OpRewritePattern<mlir::ONNXSliceOp> {
  using OpRewritePattern<mlir::ONNXSliceOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXSliceOp sliceOp,
      mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = sliceOp.getLoc();

    // Check if input is from a Conv operation
    auto convOp = sliceOp.getData().getDefiningOp<mlir::ONNXConvOp>();
    if (!convOp)
      return mlir::failure();
    // Constraint: Conv must be non-grouped (group=1)
    if (!isNonGroupedConv(convOp))
      return mlir::failure();
    // Constraint: Conv output must have single use (the slice)
    if (!convOp.getY().hasOneUse())
      return mlir::failure();

    // Get shapes
    auto convOutputShape = getShape(convOp.getY());
    auto sliceOutputShape = getShape(sliceOp.getOutput());

    if (convOutputShape.empty() || sliceOutputShape.empty())
      return mlir::failure();
    // Must be 4D (NCHW) or 5D (NCDHW) format
    if (convOutputShape.size() != 4 && convOutputShape.size() != 5)
      return mlir::failure();

    int64_t rank = convOutputShape.size();

    // Extract slice parameters
    // Assumes slice is already in standardized format (from
    // StandardizeSliceOpsPass):
    // - starts, ends, steps are dense tensors covering all dimensions
    // - axes is [0, 1, 2, ..., rank-1]
    // - No negative indices (already resolved)
    llvm::SmallVector<int64_t> fullStarts;
    llvm::SmallVector<int64_t> fullEnds;
    llvm::SmallVector<int64_t> fullSteps;

    if (!getConstantIntArray(sliceOp.getStarts(), fullStarts))
      return mlir::failure();
    if (!getConstantIntArray(sliceOp.getEnds(), fullEnds))
      return mlir::failure();
    if (!getConstantIntArray(sliceOp.getSteps(), fullSteps))
      return mlir::failure();

    // Verify slice is in standardized format (covers all dimensions)
    if (fullStarts.size() != static_cast<size_t>(rank) ||
        fullEnds.size() != static_cast<size_t>(rank) ||
        fullSteps.size() != static_cast<size_t>(rank))
      return mlir::failure();

    // Constraint: Batch dimension must be unchanged
    if (!batchUnchanged(convOutputShape, sliceOutputShape, fullSteps))
      return mlir::failure();
    // Constraint: Spatial strides must be 1 (no striding on H/W)
    if (!spatialStridesAreOne(fullSteps))
      return mlir::failure();
    // Constraint: Slice input and output quantization parameters must match
    // (if quantized). This ensures the transformation is numerically correct.
    if (!sliceQuantParamsMatch(sliceOp)) {
      std::cout << "Slice quant params don't match - skipping transformation"
                << std::endl;
      return mlir::failure();
    }
    // Check for spatial (H/W/D) slice
    bool isSpatialSlice =
        hasSpatialSlice(convOutputShape, sliceOutputShape, fullSteps);

    // Constraint: Conv3D (5D) with spatial slice is not supported
    if (isSpatialSlice && rank == 5) {
      // Conv3D + spatial slice not implemented
      return mlir::failure();
    }

    // Get channel slice parameters (channel is at index 1 in NCHW)
    int64_t channelBegin = fullStarts[1];
    int64_t channelEnd = fullEnds[1];
    int64_t channelStride = fullSteps[1];

    // Check if there's anything to optimize
    bool hasChannelSlice = channelBegin != 0 ||
                           channelEnd != convOutputShape[1] ||
                           channelStride != 1;

    if (!hasChannelSlice && !isSpatialSlice)
      return mlir::failure();
    // Get conv input for spatial slice processing
    mlir::Value convInput = convOp.getX();
    auto convInputShape = getShape(convInput);

    // Get conv attributes
    llvm::SmallVector<int64_t> strides;
    llvm::SmallVector<int64_t> pads;
    llvm::SmallVector<int64_t> dilations;
    llvm::SmallVector<int64_t> kernelShape;
    if (!getConvAttributes(convOp, strides, pads, dilations, kernelShape))
      return mlir::failure();

    // New padding values (may be updated for spatial slice)
    llvm::SmallVector<int64_t> newPads = pads;

    //===--------------------------------------------------------------------===//
    // Handle Spatial Slice: Move slice before conv, adjust padding
    //===--------------------------------------------------------------------===//
    if (isSpatialSlice) {
      // Get spatial slice parameters (H is at index 2, W is at index 3 in NCHW)
      llvm::SmallVector<int64_t> outputBeginHW = {fullStarts[2], fullStarts[3]};
      llvm::SmallVector<int64_t> outputEndHW = {fullEnds[2], fullEnds[3]};

      // Calculate which input region we need
      auto spatialParams = calculateSpatialSliceParams(convInputShape,
          outputBeginHW, outputEndHW, strides, pads, dilations, kernelShape);

      // Create slice on conv input to select the required region
      // Slice format: [N_start, C_start, H_start, W_start], [N_end, C_end,
      // H_end, W_end]
      llvm::SmallVector<int64_t> inputSliceStarts = {0, // N - keep all batches
          0, // C - keep all input channels
          spatialParams.inputBeginH, spatialParams.inputBeginW};
      llvm::SmallVector<int64_t> inputSliceEnds = {convInputShape[0], // N
          convInputShape[1],                                          // C
          spatialParams.inputEndH, spatialParams.inputEndW};
      llvm::SmallVector<int64_t> inputSliceAxes = {0, 1, 2, 3};
      llvm::SmallVector<int64_t> inputSliceSteps = {1, 1, 1, 1};

      // Only create slice if we're actually cropping
      if (spatialParams.inputBeginH != 0 ||
          spatialParams.inputEndH != convInputShape[2] ||
          spatialParams.inputBeginW != 0 ||
          spatialParams.inputEndW != convInputShape[3]) {

        // Create slice using convOp.getY().getType() for encoding
        // (similar to how gather->slice uses gatherOp.getType())
        convInput = createSliceOp(rewriter, loc, convInput, inputSliceStarts,
            inputSliceEnds, inputSliceAxes, inputSliceSteps,
            mlir::cast<mlir::RankedTensorType>(convOp.getY().getType()));
      }

      // Update padding for the new conv
      newPads = {spatialParams.newPadTop, spatialParams.newPadLeft,
          spatialParams.newPadBottom, spatialParams.newPadRight};
    }

    //===--------------------------------------------------------------------===//
    // Handle Channel Slice: Slice weights and bias
    //===--------------------------------------------------------------------===//

    // Get weights
    mlir::Value weights = convOp.getW();
    auto weightConstOp = weights.getDefiningOp<mlir::ONNXConstantOp>();
    if (!weightConstOp)
      return mlir::failure();

    auto weightShape = getShape(weights);
    if (weightShape.empty())
      return mlir::failure();

    // Check if weights are quantized
    bool weightsAreQuantized = isQuantizedType(weights.getType());

    mlir::Value newWeights = weights;
    mlir::Value newBias = convOp.getB();
    int64_t newOutputChannels = convOutputShape[1];

    if (hasChannelSlice) {

      // Calculate new weight shape
      newOutputChannels =
          calculateSlicedChannels(channelBegin, channelEnd, channelStride);

      llvm::SmallVector<int64_t> newWeightShape = weightShape;
      newWeightShape[0] = newOutputChannels;

      if (weightsAreQuantized) {
        // Handle quantized weights (int8/uint8 with quant.uniform type)
        llvm::SmallVector<int64_t> weightDataInt;
        if (!getConstantIntData(weightConstOp, weightDataInt))
          return mlir::failure();

        // Slice integer weight data
        auto slicedWeightData = sliceIntWeightData(weightDataInt, weightShape,
            channelBegin, channelEnd, channelStride);

        // Create new quantized weights constant (preserves quantization type)
        newWeights = createConstantFromIntData(rewriter, loc, newWeightShape,
            slicedWeightData, getElementType(weights));
      } else {
        // Handle float weights
        llvm::SmallVector<float> weightData;
        if (!getConstantFloatData(weightConstOp, weightData))
          return mlir::failure();

        // Slice float weight data
        auto slicedWeightData = sliceWeightData(
            weightData, weightShape, channelBegin, channelEnd, channelStride);

        // Create new float weights constant
        newWeights = createConstantFromFloatData(rewriter, loc, newWeightShape,
            slicedWeightData, getElementType(weights));
      }

      // Handle bias if present
      if (convOp.getB() &&
          !mlir::isa<mlir::NoneType>(convOp.getB().getType())) {
        auto biasConstOp = convOp.getB().getDefiningOp<mlir::ONNXConstantOp>();
        if (biasConstOp) {
          llvm::SmallVector<int64_t> newBiasShape = {newOutputChannels};
          bool biasIsQuantized = isQuantizedType(convOp.getB().getType());

          if (biasIsQuantized) {
            // Handle quantized bias (int32 with quant.uniform type)
            llvm::SmallVector<int64_t> biasDataInt;
            if (getConstantIntData(biasConstOp, biasDataInt)) {
              auto slicedBiasData = sliceIntBiasData(
                  biasDataInt, channelBegin, channelEnd, channelStride);

              newBias = createConstantFromIntData(rewriter, loc, newBiasShape,
                  slicedBiasData, getElementType(convOp.getB()));
            }
          } else {
            // Handle float bias
            llvm::SmallVector<float> biasData;
            if (getConstantFloatData(biasConstOp, biasData)) {
              auto slicedBiasData = sliceBiasData(
                  biasData, channelBegin, channelEnd, channelStride);

              newBias = createConstantFromFloatData(rewriter, loc, newBiasShape,
                  slicedBiasData, getElementType(convOp.getB()));
            }
          }
        }
      }
    }

    //===--------------------------------------------------------------------===//
    // Create new Conv operation
    //===--------------------------------------------------------------------===//

    // Compute new conv output shape
    llvm::SmallVector<int64_t> newConvOutputShape = sliceOutputShape;

    // Use the original slice output's element type to ensure compatibility with
    // downstream ops (e.g., QuantizeLinear expects float, not quantized type).
    // The slice may have converted from quantized to float implicitly.
    auto newConvOutputType = mlir::RankedTensorType::get(
        newConvOutputShape, getElementType(sliceOp.getOutput()));

    // Create new pads attribute
    auto newPadsAttr = rewriter.getI64ArrayAttr(newPads);

    // Create new Conv operation
    auto newConvOp = rewriter.create<mlir::ONNXConvOp>(loc, newConvOutputType,
        convInput,  // Possibly sliced input (for spatial slice)
        newWeights, // Possibly sliced weights (for channel slice)
        newBias,    // Possibly sliced bias (for channel slice)
        convOp.getAutoPadAttr(), convOp.getDilationsAttr(),
        convOp.getGroupAttr(), convOp.getKernelShapeAttr(),
        newPadsAttr, // Adjusted padding (for spatial slice)
        convOp.getStridesAttr());

    // Replace the slice output with the new conv output
    rewriter.replaceOp(sliceOp, newConvOp.getY());

    // The original convOp will be removed by DCE since it has no uses

    return mlir::success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct TransferConvSliceToConvPass
    : public PassWrapper<TransferConvSliceToConvPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transfer-conv-slice-to-conv";
  }
  StringRef getDescription() const override {
    return "Transfer Conv -> Slice patterns by moving slice before convolution";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<TransferConvSliceToConvPattern>(context);

    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createTransferConvSliceToConvPass() {
  return std::make_unique<TransferConvSliceToConvPass>();
}

} // namespace onnx_mlir
