// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "llvm/ADT/SmallVector.h"

#include <cmath>
#include <numeric>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get shape from a value
llvm::SmallVector<int64_t> getShape(mlir::Value value) {
  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasRank())
    return {};
  return llvm::SmallVector<int64_t>(
      shapedType.getShape().begin(), shapedType.getShape().end());
}

/// Check if a number is power of 2
bool isPowerOf2(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

float getMulConstCoefficient(int64_t channels) {

  return 1.0f / static_cast<float>(channels);
}

/// Normalize axis to positive value
int64_t normalizeAxis(int64_t axis, int64_t rank) {
  return axis < 0 ? axis + rank : axis;
}

/// Check if reduction is only on channel dimension (axis 1 for NCHW)
bool isChannelWiseReduction(llvm::ArrayRef<int64_t> axes, int64_t rank) {
  if (axes.size() != 1)
    return false;
  int64_t axis = normalizeAxis(axes[0], rank);
  return axis == 1; // NCHW: channel is at index 1
}

/// Create constant tensor with given shape and value.
/// Supports float, integer, and quantized element types.
/// For quantized types, the float value is quantized using the type's
/// scale and zero_point, and stored using the integer storage type.
mlir::Value createConstantTensor(mlir::PatternRewriter &rewriter,
    mlir::Location loc, llvm::ArrayRef<int64_t> shape, float value,
    mlir::Type elementType) {
  int64_t numElements = std::accumulate(
      shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  // Determine storage type (for quantized types, this is the underlying int)
  mlir::Type storageType = elementType;
  if (auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(elementType))
    storageType = quantType.getStorageType();

  // Result type uses the full element type (including quantization info)
  auto resultType = mlir::RankedTensorType::get(shape, elementType);
  // Storage tensor type uses the storage type for DenseElementsAttr
  auto storageTensorType = mlir::RankedTensorType::get(shape, storageType);

  mlir::Attribute denseAttr;

  if (mlir::isa<mlir::FloatType>(storageType)) {
    // Float path
    llvm::SmallVector<float> values(numElements, value);
    denseAttr = mlir::DenseElementsAttr::get(
        storageTensorType, llvm::ArrayRef<float>(values));
  } else {
    // Integer/quantized path
    int64_t intValue = static_cast<int64_t>(std::round(value));

    // For quantized types, convert float value to quantized integer
    if (auto uniformQType =
            mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elementType)) {
      double scale = uniformQType.getScale();
      int64_t zp = uniformQType.getZeroPoint();
      intValue = static_cast<int64_t>(std::round(value / scale)) + zp;
      // Clamp to storage range
      intValue =
          std::max(intValue, static_cast<int64_t>(uniformQType.getStorageTypeMin()));
      intValue =
          std::min(intValue, static_cast<int64_t>(uniformQType.getStorageTypeMax()));
    }

    unsigned bitWidth = storageType.getIntOrFloatBitWidth();
    // Determine signedness: signless/signed integers and signed quantized types
    // use signed representation; unsigned integers and unsigned quantized types
    // use unsigned representation.
    bool isSigned = !storageType.isUnsignedInteger();
    llvm::SmallVector<mlir::APInt> apIntData;
    apIntData.reserve(numElements);
    for (int64_t i = 0; i < numElements; ++i)
      apIntData.push_back(
          mlir::APInt(bitWidth, static_cast<uint64_t>(intValue), isSigned));

    denseAttr = mlir::DenseIntElementsAttr::get(storageTensorType, apIntData);
  }

  return rewriter.create<mlir::ONNXConstantOp>(loc, resultType,
      mlir::ValueRange{},
      mlir::ArrayRef<mlir::NamedAttribute>{
          rewriter.getNamedAttr("value", denseAttr)});
}

/// Create reshape operation
mlir::Value createReshape(mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, llvm::ArrayRef<int64_t> targetShape) {
  auto inputType = mlir::cast<mlir::ShapedType>(input.getType());
  auto targetType =
      mlir::RankedTensorType::get(targetShape, inputType.getElementType());

  // Create shape constant
  auto shapeType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(targetShape.size())}, rewriter.getI64Type());
  auto shapeAttr = mlir::DenseIntElementsAttr::get(shapeType, targetShape);
  auto shapeConst =
      rewriter.create<mlir::ONNXConstantOp>(loc, shapeType, mlir::ValueRange{},
          mlir::ArrayRef<mlir::NamedAttribute>{
              rewriter.getNamedAttr("value", shapeAttr)});

  return rewriter.create<mlir::ONNXReshapeOp>(
      loc, targetType, input, shapeConst, /*allowzero=*/0);
}

/// Compute 4D shape for convolution input (NCHW layout)
llvm::SmallVector<int64_t> compute4DShape(llvm::ArrayRef<int64_t> inputShape) {
  llvm::SmallVector<int64_t> result;

  if (inputShape.size() < 4) {
    // For NCHW: [N, C, ...] -> [N, C, 1, 1] or similar
    // Pad with 1s at the end for spatial dimensions
    result.assign(inputShape.begin(), inputShape.end());
    result.resize(4, 1);
  } else if (inputShape.size() == 4) {
    result.assign(inputShape.begin(), inputShape.end());
  } else {
    // Flatten spatial dimensions: [N, C, D1, D2, ..., Dk] -> [N, C, 1, W]
    int64_t n = inputShape[0];
    int64_t c = inputShape[1]; // NCHW: channel is at index 1
    int64_t w = std::accumulate(std::next(inputShape.begin(), 2),
        inputShape.end(), 1LL, std::multiplies<int64_t>());
    result = {n, c, 1, w};
  }

  return result;
}

/// Convert channel-wise reduction to convolution (NCHW layout)
mlir::Value convertReductionToConv(mlir::PatternRewriter &rewriter,
    mlir::Location loc, mlir::Value input,
    bool isMean, // true for mean, false for sum
    llvm::ArrayRef<int64_t> originalOutputShape) {

  auto inputShape = getShape(input);
  int64_t rank = inputShape.size();
  int64_t inputChannel = inputShape[1]; // NCHW: channel is at index 1

  // For Conv3D, keep 5D; otherwise reshape to 4D
  bool useConv3D = (rank == 5);

  llvm::SmallVector<int64_t> convInputShape;
  if (useConv3D) {
    // Keep 5D for Conv3D: [N, C, D, H, W]
    convInputShape = inputShape;
  } else {
    convInputShape = compute4DShape(inputShape);
  }

  // Reshape if needed
  mlir::Value convInput = input;
  bool needsInputReshape =
      (inputShape != llvm::ArrayRef<int64_t>(convInputShape));
  if (needsInputReshape) {
    convInput = createReshape(rewriter, loc, input, convInputShape);
  }

  // NCHW layout: standard ONNX Conv format

  int64_t convOutputC = 1; // Channel-wise reduction always outputs 1 channel

  float weightValue = isMean ? (1.0f / static_cast<float>(inputChannel)) : 1.0f;

  llvm::SmallVector<int64_t> weightShape;
  llvm::SmallVector<int64_t> kernelShape;
  llvm::SmallVector<int64_t> strides;
  llvm::SmallVector<int64_t> pads;
  llvm::SmallVector<int64_t> dilations;

  if (useConv3D) {
    // Conv3D weight: [O, I, D, H, W] for NCHW
    weightShape = {convOutputC, inputChannel, 1, 1, 1};
    kernelShape = {1, 1, 1};
    strides = {1, 1, 1};
    pads = {0, 0, 0, 0, 0, 0};
    dilations = {1, 1, 1};
  } else {
    // Conv2D weight: [O, I, H, W] for NCHW
    weightShape = {convOutputC, inputChannel, 1, 1};
    kernelShape = {1, 1};
    strides = {1, 1};
    pads = {0, 0, 0, 0};
    dilations = {1, 1};
  }

  // Create weights
  auto inputType = mlir::cast<mlir::ShapedType>(convInput.getType());
  mlir::Value weights = createConstantTensor(
      rewriter, loc, weightShape, weightValue, inputType.getElementType());

  // Compute output shape (same spatial dims, channel becomes 1)
  llvm::SmallVector<int64_t> convOutputShape;
  if (useConv3D) {
    // [N, C, D, H, W] → [N, 1, D, H, W]
    convOutputShape = {convInputShape[0], convOutputC, convInputShape[2],
        convInputShape[3], convInputShape[4]};
  } else {
    // [N, C, H, W] → [N, 1, H, W]
    convOutputShape = {
        convInputShape[0], convOutputC, convInputShape[2], convInputShape[3]};
  }

  auto convOutputType =
      mlir::RankedTensorType::get(convOutputShape, inputType.getElementType());

  // Create None value for bias (required by ONNX dialect)
  auto noneVal = rewriter.create<mlir::ONNXNoneOp>(loc).getResult();

  // Create signed 64-bit integer type for group attribute
  auto si64Type = rewriter.getIntegerType(64, /*isSigned=*/true);

  // Create Conv operation
  mlir::Value convResult =
      rewriter.create<mlir::ONNXConvOp>(loc, convOutputType, convInput, weights,
          /*B=*/noneVal, // no bias
          /*auto_pad=*/rewriter.getStringAttr("NOTSET"),
          /*dilations=*/rewriter.getI64ArrayAttr(dilations),
          /*group=*/mlir::IntegerAttr::get(si64Type, 1),
          /*kernel_shape=*/rewriter.getI64ArrayAttr(kernelShape),
          /*pads=*/rewriter.getI64ArrayAttr(pads),
          /*strides=*/rewriter.getI64ArrayAttr(strides));

  // Reshape back to original output shape if needed
  auto convResultShape = getShape(convResult);
  if (convResultShape != llvm::ArrayRef<int64_t>(originalOutputShape)) {
    convResult = createReshape(rewriter, loc, convResult, originalOutputShape);
  }

  return convResult;
}

/// Get reduction axes from ReduceMean op
llvm::SmallVector<int64_t> getAxesFromReduceMean(mlir::ONNXReduceMeanOp op) {
  llvm::SmallVector<int64_t> axes;
  mlir::Value axesInput = op.getAxes();
  // Try to get axes from constant input
  if (auto defOp = axesInput.getDefiningOp<mlir::ONNXConstantOp>()) {
    if (auto valueAttr = defOp.getValueAttr()) {
      if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr)) {
        for (auto val : denseAttr.getValues<mlir::APInt>())
          axes.push_back(val.getSExtValue());
      }
    }
  }
  return axes;
}

/// Get reduction axes from ReduceSum op
llvm::SmallVector<int64_t> getAxesFromReduceSum(mlir::Value axesInput) {
  llvm::SmallVector<int64_t> axes;

  // Try to get axes from constant input
  if (auto defOp = axesInput.getDefiningOp<mlir::ONNXConstantOp>()) {
    if (auto valueAttr = defOp.getValueAttr()) {
      if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr)) {
        for (auto val : denseAttr.getValues<mlir::APInt>())
          axes.push_back(val.getSExtValue());
      }
    }
  }

  return axes;
}

//===----------------------------------------------------------------------===//
// ReduceMeanToConvPattern
//===----------------------------------------------------------------------===//

struct ReduceMeanToConvPattern : public OpRewritePattern<ONNXReduceMeanOp> {
  using OpRewritePattern<ONNXReduceMeanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReduceMeanOp op, PatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();
    mlir::Value input = op.getData();

    // Get input shape
    auto inputShape = getShape(input);
    if (inputShape.empty() || inputShape.size() < 2)
      return mlir::failure();

    int64_t rank = inputShape.size();
    int64_t inputChannel = inputShape[1]; // NCHW: channel is at index 1

    // Get reduction axes
    auto axes = getAxesFromReduceMean(op);
    if (axes.empty())
      return mlir::failure();

    // Check if channel-wise reduction
    if (!isChannelWiseReduction(axes, rank)) {
      // Not a channel-wise reduction, skip
      return mlir::failure();
    }

    // For reduction_mean, channel must be power of 2
    // (unless this is part of a mul pattern - handled separately)
    if (!isPowerOf2(inputChannel)) {
      // Cannot convert to conv - channel is not power of 2
      return mlir::failure();
    }

    // Check output shape constraint
    // NCHW: channel at index 1, so valid output has channel dim reduced
    auto outputShape = getShape(op.getReduced());
    bool validOutputShape =
        (outputShape.size() == inputShape.size() - 1) ||
        (outputShape.size() == inputShape.size() && outputShape[1] == 1);

    if (!validOutputShape)
      return mlir::failure();

    // Convert to convolution
    mlir::Value result = convertReductionToConv(rewriter, loc, input,
        /*isMean=*/true, outputShape);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReduceSumToConvPattern
//===----------------------------------------------------------------------===//

struct ReduceSumToConvPattern : public OpRewritePattern<ONNXReduceSumOp> {
  using OpRewritePattern<ONNXReduceSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReduceSumOp op, PatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();
    mlir::Value input = op.getData();
    mlir::Value axesInput = op.getAxes();

    // Get input shape
    auto inputShape = getShape(input);
    if (inputShape.empty() || inputShape.size() < 2)
      return mlir::failure();

    int64_t rank = inputShape.size();

    // Get reduction axes from constant input
    auto axes = getAxesFromReduceSum(axesInput);
    if (axes.empty())
      return mlir::failure();

    // Check if channel-wise reduction
    if (!isChannelWiseReduction(axes, rank))
      return mlir::failure();

    // Check output shape constraint
    // NCHW: channel at index 1, so valid output has channel dim reduced
    auto outputShape = getShape(op.getReduced());
    bool validOutputShape =
        (outputShape.size() == inputShape.size() - 1) ||
        (outputShape.size() == inputShape.size() && outputShape[1] == 1);

    if (!validOutputShape)
      return mlir::failure();

    // Convert to convolution
    mlir::Value result = convertReductionToConv(rewriter, loc, input,
        /*isMean=*/false, outputShape);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReduceMeanMulToConvPattern
//===----------------------------------------------------------------------===//

struct ReduceMeanMulToConvPattern : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const override {

    // Check if one input is ReduceMean
    auto *lhsDef = mulOp.getA().getDefiningOp();
    auto *rhsDef = mulOp.getB().getDefiningOp();

    auto lhsReduce = mlir::dyn_cast_or_null<mlir::ONNXReduceMeanOp>(lhsDef);
    auto rhsReduce = mlir::dyn_cast_or_null<mlir::ONNXReduceMeanOp>(rhsDef);

    if (!lhsReduce && !rhsReduce)
      return mlir::failure();

    auto reduceMean = lhsReduce ? lhsReduce : rhsReduce;
    mlir::Value constInput = lhsReduce ? mulOp.getB() : mulOp.getA();

    // Check single use
    if (!reduceMean.getReduced().hasOneUse())
      return mlir::failure();

    // Get constant value
    auto constOp = constInput.getDefiningOp<mlir::ONNXConstantOp>();
    if (!constOp)
      return mlir::failure();

    auto valueAttr = constOp.getValueAttr();
    if (!valueAttr)
      return mlir::failure();

    auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr);
    if (!denseAttr || !denseAttr.isSplat())
      return mlir::failure();

    // Extract the constant multiplier value as float
    float mulConstant;
    mlir::Type constElemType = denseAttr.getElementType();
    if (mlir::isa<mlir::FloatType>(constElemType)) {
      mulConstant =
          denseAttr.getSplatValue<mlir::APFloat>().convertToFloat();
    } else if (constElemType.isIntOrIndex()) {
      mulConstant = static_cast<float>(
          denseAttr.getSplatValue<mlir::APInt>().getSExtValue());
    } else {
      // Unsupported element type for constant extraction
      return mlir::failure();
    }

    // Get input info
    mlir::Value input = reduceMean.getData();
    auto inputShape = getShape(input);
    if (inputShape.empty() || inputShape.size() < 2)
      return mlir::failure();

    int64_t rank = inputShape.size();
    int64_t inputChannel = inputShape[1]; // NCHW: channel is at index 1

    // Check channel-wise reduction
    auto axes = getAxesFromReduceMean(reduceMean);
    if (!isChannelWiseReduction(axes, rank))
      return mlir::failure();

    // Validate: Does mulConstant == 1/inputChannel (or DPU approximation)?
    float expectedCoeff = getMulConstCoefficient(inputChannel);
    float tolerance = 1e-6f;
    if (std::abs(mulConstant - expectedCoeff) > tolerance) {
      // Constant doesn't match expected DPU coefficient
      return mlir::failure();
    }

    // Get output shape from mul op
    auto outputShape = getShape(mulOp.getC());

    // Convert to convolution (the mul is absorbed into the conv weights)
    mlir::Location loc = mulOp.getLoc();
    mlir::Value result = convertReductionToConv(rewriter, loc, input,
        /*isMean=*/true, outputShape);

    rewriter.replaceOp(mulOp, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReduceMeanReluToConvPattern
//===----------------------------------------------------------------------===//

struct ReduceMeanReluToConvPattern : public OpRewritePattern<ONNXReluOp> {
  using OpRewritePattern<ONNXReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReluOp reluOp, PatternRewriter &rewriter) const override {

    // Check if input is ReduceMean
    auto reduceMean = reluOp.getX().getDefiningOp<mlir::ONNXReduceMeanOp>();
    if (!reduceMean)
      return mlir::failure();

    // Check single use of reduceMean
    if (!reduceMean.getReduced().hasOneUse())
      return mlir::failure();

    // Get input info
    mlir::Value input = reduceMean.getData();
    auto inputShape = getShape(input);
    if (inputShape.empty() || inputShape.size() < 2)
      return mlir::failure();

    int64_t rank = inputShape.size();
    int64_t inputChannel = inputShape[1]; // NCHW: channel is at index 1

    // Check channel-wise reduction
    auto axes = getAxesFromReduceMean(reduceMean);
    if (!isChannelWiseReduction(axes, rank))
      return mlir::failure();

    // Power of 2 constraint
    if (!isPowerOf2(inputChannel))
      return mlir::failure();

    // Get output shape from relu
    auto reluOutputShape = getShape(reluOp.getY());
    auto reduceMeanOutputShape = getShape(reduceMean.getReduced());

    // Convert reduction to conv
    mlir::Location loc = reluOp.getLoc();
    mlir::Value convResult = convertReductionToConv(
        rewriter, loc, input, /*isMean=*/true, reduceMeanOutputShape);

    // Create new ReLU on conv output
    auto reluOutputType = mlir::RankedTensorType::get(reluOutputShape,
        mlir::cast<mlir::ShapedType>(convResult.getType()).getElementType());

    auto newRelu =
        rewriter.create<mlir::ONNXReluOp>(loc, reluOutputType, convResult);

    rewriter.replaceOp(reluOp, newRelu.getY());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct TransferReduceMeanSumToConvPass
    : public PassWrapper<TransferReduceMeanSumToConvPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transfer-reduce-mean-sum-to-conv";
  }
  StringRef getDescription() const override {
    return "Transfer ReduceMean and ReduceSum operations to Conv operations";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReduceMeanMulToConvPattern>(context);
    patterns.add<ReduceMeanReluToConvPattern>(context);
    patterns.add<ReduceMeanToConvPattern>(context);
    patterns.add<ReduceSumToConvPattern>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createTransferReduceMeanSumToConvPass() {
  return std::make_unique<TransferReduceMeanSumToConvPass>();
}

} // namespace onnx_mlir
