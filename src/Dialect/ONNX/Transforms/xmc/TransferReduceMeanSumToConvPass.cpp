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
#include <numeric>
#include <type_traits>

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

/// Check if reduction is on channel dimension.
/// NCHW: channel at axis 1. NHWC: channel at axis rank-1.
bool isChannelWiseReduction(llvm::ArrayRef<int64_t> axes, int64_t rank) {
  if (axes.size() != 1)
    return false;
  int64_t axis = normalizeAxis(axes[0], rank);
  return axis == 1 || (rank > 2 && axis == rank - 1);
}

/// Check if element type is valid for onnx.Conv (float or quantized).
/// Integer types like i64 (from Shape/Cast ops) are not supported by Conv.
bool isConvCompatibleElementType(mlir::Type elementType) {
  return mlir::isa<mlir::FloatType>(elementType) ||
         mlir::isa<mlir::quant::QuantizedType>(elementType);
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
      intValue = std::max(
          intValue, static_cast<int64_t>(uniformQType.getStorageTypeMin()));
      intValue = std::min(
          intValue, static_cast<int64_t>(uniformQType.getStorageTypeMax()));
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

/// Convert channel-wise reduction to convolution (NCHW layout).
/// Standard group=1 1x1 conv reduces inputShape[1] (C_in) to 1.
///   Input [N,C,H,W] → Conv(weight=[1,C,1,1], group=1) → [N,1,H,W]
/// The downstream ONNXToXIR pass handles NCHW→NHWC layout conversion.
/// \p outputElementType is the element type of the original reduction op's
/// output, used for the Conv result so downstream ops see the correct type.
mlir::Value convertReductionToConv(mlir::PatternRewriter &rewriter,
    mlir::Location loc, mlir::Value input,
    bool isMean, // true for mean, false for sum
    llvm::ArrayRef<int64_t> originalOutputShape, mlir::Type outputElementType) {

  auto inputShape = getShape(input);
  int64_t rank = inputShape.size();
  int64_t inputChannel = inputShape[1]; // NCHW: channel is at index 1

  // For Conv3D, keep 5D; otherwise reshape to 4D
  bool useConv3D = (rank == 5);

  llvm::SmallVector<int64_t> convInputShape;
  if (useConv3D) {
    convInputShape = inputShape;
  } else {
    convInputShape = compute4DShape(inputShape);
  }

  mlir::Value convInput = input;
  if (inputShape != llvm::ArrayRef<int64_t>(convInputShape)) {
    convInput = createReshape(rewriter, loc, input, convInputShape);
  }

  int64_t convOutputC = 1;
  float weightValue = isMean ? (1.0f / static_cast<float>(inputChannel)) : 1.0f;

  llvm::SmallVector<int64_t> weightShape;
  llvm::SmallVector<int64_t> kernelShape;
  llvm::SmallVector<int64_t> strides;
  llvm::SmallVector<int64_t> pads;
  llvm::SmallVector<int64_t> dilations;

  if (useConv3D) {
    weightShape = {convOutputC, inputChannel, 1, 1, 1};
    kernelShape = {1, 1, 1};
    strides = {1, 1, 1};
    pads = {0, 0, 0, 0, 0, 0};
    dilations = {1, 1, 1};
  } else {
    weightShape = {convOutputC, inputChannel, 1, 1};
    kernelShape = {1, 1};
    strides = {1, 1};
    pads = {0, 0, 0, 0};
    dilations = {1, 1};
  }

  auto inputType = mlir::cast<mlir::ShapedType>(convInput.getType());

  // For quantized inputs, compute weight scale via fix_point:
  //   fix_point = floor(log2(127 / |weightValue|))
  //   wScale    = 1 / 2^fix_point
  // This maximizes INT8 utilization (quantized weight ≈ 64 for power-of-2 C).
  mlir::Type weightElemType = inputType.getElementType();
  double wScale = 1.0;
  if (auto inputQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
          inputType.getElementType())) {
    double absWeight = std::abs(static_cast<double>(weightValue));
    if (absWeight > 1e-10) {
      int fixPoint = static_cast<int>(std::floor(std::log2(127.0 / absWeight)));
      wScale = 1.0 / std::pow(2.0, fixPoint);
    }
    weightElemType = mlir::quant::UniformQuantizedType::get(
        mlir::quant::QuantizationFlags::Signed, rewriter.getIntegerType(8),
        inputQuantType.getExpressedType(), wScale,
        /*zeroPoint=*/0, /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  }

  mlir::Value weights = createConstantTensor(
      rewriter, loc, weightShape, weightValue, weightElemType);

  llvm::SmallVector<int64_t> convOutputShape;
  if (useConv3D) {
    convOutputShape = {convInputShape[0], convOutputC, convInputShape[2],
        convInputShape[3], convInputShape[4]};
  } else {
    convOutputShape = {
        convInputShape[0], convOutputC, convInputShape[2], convInputShape[3]};
  }

  auto convOutputType =
      mlir::RankedTensorType::get(convOutputShape, outputElementType);

  mlir::Value bias;
  if (auto inputQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
          inputType.getElementType())) {
    llvm::SmallVector<int64_t> biasShape = {convOutputC};
    double biasScale = inputQuantType.getScale() * wScale;
    auto biasQuantType = mlir::quant::UniformQuantizedType::get(
        mlir::quant::QuantizationFlags::Signed, rewriter.getIntegerType(16),
        inputQuantType.getExpressedType(), biasScale,
        /*zeroPoint=*/0, /*storageTypeMin=*/-32768, /*storageTypeMax=*/32767);
    bias = createConstantTensor(rewriter, loc, biasShape, 0.0f, biasQuantType);
  } else {
    bias = rewriter.create<mlir::ONNXNoneOp>(loc).getResult();
  }

  auto si64Type = rewriter.getIntegerType(64, /*isSigned=*/true);

  mlir::Value convResult =
      rewriter.create<mlir::ONNXConvOp>(loc, convOutputType, convInput, weights,
          /*B=*/bias,
          /*auto_pad=*/rewriter.getStringAttr("NOTSET"),
          /*dilations=*/rewriter.getI64ArrayAttr(dilations),
          /*group=*/mlir::IntegerAttr::get(si64Type, 1),
          /*kernel_shape=*/rewriter.getI64ArrayAttr(kernelShape),
          /*pads=*/rewriter.getI64ArrayAttr(pads),
          /*strides=*/rewriter.getI64ArrayAttr(strides));

  // Reshape to original output shape if needed (e.g. keepdims=false)
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

/// Get reduction axes from ReduceMeanV13 op (axes are a ArrayAttr, not an SSA
/// operand).
llvm::SmallVector<int64_t> getAxesFromReduceMeanV13(
    mlir::ONNXReduceMeanV13Op op) {
  llvm::SmallVector<int64_t> axes;
  mlir::ArrayAttr arrayAttr = op.getAxesAttr();
  if (!arrayAttr)
    return axes;
  for (mlir::Attribute elem : arrayAttr) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(elem))
      axes.push_back(intAttr.getInt());
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

template <typename ReduceMeanOpTy>
struct ReduceMeanToConvPattern : public OpRewritePattern<ReduceMeanOpTy> {
  using OpRewritePattern<ReduceMeanOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ReduceMeanOpTy op, PatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();
    mlir::Value input = op.getData();

    auto inputShape = getShape(input);
    if (inputShape.empty() || inputShape.size() < 2)
      return mlir::failure();

    if (!op.getReduced().hasOneUse())
      return mlir::failure();

    // onnx.Conv only supports float/quantized types, not integer types.
    auto inputElemType =
        mlir::cast<mlir::ShapedType>(input.getType()).getElementType();
    if (!isConvCompatibleElementType(inputElemType))
      return mlir::failure();

    int64_t rank = inputShape.size();

    llvm::SmallVector<int64_t> axes;
    if constexpr (std::is_same_v<ReduceMeanOpTy, mlir::ONNXReduceMeanOp>) {
      axes = getAxesFromReduceMean(op);
    } else {
      static_assert(std::is_same_v<ReduceMeanOpTy, mlir::ONNXReduceMeanV13Op>,
          "ReduceMeanToConvPattern only supports ReduceMean and ReduceMeanV13");
      axes = getAxesFromReduceMeanV13(op);
    }
    if (axes.empty())
      return mlir::failure();

    if (!isChannelWiseReduction(axes, rank))
      return mlir::failure();

    int64_t inputChannel = inputShape[1];

    if (!isPowerOf2(inputChannel))
      return mlir::failure();

    auto outputShape = getShape(op.getReduced());
    bool validOutputShape =
        (outputShape.size() == inputShape.size() - 1) ||
        (outputShape.size() == inputShape.size() && outputShape[1] == 1);

    if (!validOutputShape)
      return mlir::failure();

    auto outputElemType =
        mlir::cast<mlir::ShapedType>(op.getReduced().getType())
            .getElementType();
    mlir::Value result = convertReductionToConv(rewriter, loc, input,
        /*isMean=*/true, outputShape, outputElemType);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReduceMeanSpatialAxisToConvPattern
//===----------------------------------------------------------------------===//
// Handles single-axis ReduceMean on any non-batch, non-channel axis by
// transposing the reduction axis to the channel position (axis 1), applying
// channel-wise conv reduction, then transposing back.

template <typename ReduceMeanOpTy>
struct ReduceMeanSpatialAxisToConvPattern
    : public OpRewritePattern<ReduceMeanOpTy> {
  using OpRewritePattern<ReduceMeanOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ReduceMeanOpTy op, PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value input = op.getData();

    auto inputShape = getShape(input);
    if (inputShape.empty() || inputShape.size() < 4)
      return mlir::failure();

    if (!op.getReduced().hasOneUse())
      return mlir::failure();

    int64_t rank = inputShape.size();

    llvm::SmallVector<int64_t> axes;
    if constexpr (std::is_same_v<ReduceMeanOpTy, mlir::ONNXReduceMeanOp>) {
      axes = getAxesFromReduceMean(op);
    } else {
      static_assert(std::is_same_v<ReduceMeanOpTy, mlir::ONNXReduceMeanV13Op>,
          "ReduceMeanSpatialAxisToConvPattern only supports ReduceMean and "
          "ReduceMeanV13");
      axes = getAxesFromReduceMeanV13(op);
    }
    if (axes.size() != 1)
      return mlir::failure();

    int64_t axis = normalizeAxis(axes[0], rank);

    if (axis == 0)
      return mlir::failure();

    // Only axis=1 (NCHW channel) is handled by ReduceMeanToConvPattern.
    // axis=rank-1 (NHWC channel) falls through here because
    // ReduceMeanToConvPattern hardcodes inputShape[1] as channel count,
    // which is incorrect for NHWC. This pattern correctly transposes the
    // reduction axis to position 1 before converting to Conv.
    if (axis == 1)
      return mlir::failure();

    int64_t reductionDimSize = inputShape[axis];

    if (reductionDimSize == mlir::ShapedType::kDynamic)
      return mlir::failure();

    if (!isPowerOf2(reductionDimSize))
      return mlir::failure();

    auto outputShape = getShape(op.getReduced());
    if (outputShape.empty())
      return mlir::failure();

    auto inputType = mlir::cast<mlir::ShapedType>(input.getType());
    auto outputElemType =
        mlir::cast<mlir::ShapedType>(op.getReduced().getType())
            .getElementType();

    // If rank < 4, reshape to 4D first by appending size-1 dims.
    // This ensures our transpose and ConvertToChannelLastPass's transpose
    // both operate on 4D tensors and compose to a trivial reshape
    // (only moving size-1 dims) rather than producing residual transposes.
    mlir::Value workingInput = input;
    llvm::SmallVector<int64_t> workingShape(
        inputShape.begin(), inputShape.end());
    int64_t workingRank = rank;
    if (rank < 4) {
      workingShape.resize(4, 1);
      workingInput = createReshape(rewriter, loc, input, workingShape);
      workingRank = 4;
    }

    // Build permutation: [0, axis, axis+1, ..., r-1, 1, ..., axis-1]
    // Moves reduction axis to position 1 (NCHW channel), placing dims
    // after the reduction axis first, then dims before it (excluding batch).
    // E.g. axis=3, rank=4: [0, 3, 1, 2] (standard NHWC→NCHW)
    //      axis=2, rank=4: [0, 2, 3, 1]
    // When composed with ConvertToChannelLastPass's NCHW→NHWC [0, 2, 3, 1],
    // the combined permutation only moves size-1 dims (zero-cost reshape).
    llvm::SmallVector<int64_t> perm;
    perm.push_back(0);
    perm.push_back(axis);
    for (int64_t i = axis + 1; i < workingRank; ++i)
      perm.push_back(i);
    for (int64_t i = 1; i < axis; ++i)
      perm.push_back(i);

    llvm::SmallVector<int64_t> transposedInputShape;
    for (auto p : perm)
      transposedInputShape.push_back(workingShape[p]);

    auto transposedInputType = mlir::RankedTensorType::get(
        transposedInputShape, inputType.getElementType());

    mlir::Value transposedInput = rewriter.create<mlir::ONNXTransposeOp>(
        loc, transposedInputType, workingInput, rewriter.getI64ArrayAttr(perm));

    // In transposed domain: reduction is now on axis 1 (channel)
    // Compute keepdims=true output shape in transposed domain
    llvm::SmallVector<int64_t> transposedConvOutputShape;
    for (int64_t i = 0; i < workingRank; ++i)
      transposedConvOutputShape.push_back(i == 1 ? 1 : transposedInputShape[i]);

    mlir::Value convResult =
        convertReductionToConv(rewriter, loc, transposedInput, /*isMean=*/true,
            transposedConvOutputShape, outputElemType);

    // Compute inverse permutation and transpose back
    llvm::SmallVector<int64_t> inversePerm(workingRank);
    for (int64_t i = 0; i < workingRank; ++i)
      inversePerm[perm[i]] = i;

    auto convResultShape = getShape(convResult);
    llvm::SmallVector<int64_t> untransposedShape;
    for (auto p : inversePerm)
      untransposedShape.push_back(convResultShape[p]);

    auto untransposedType =
        mlir::RankedTensorType::get(untransposedShape, outputElemType);

    mlir::Value result = rewriter.create<mlir::ONNXTransposeOp>(loc,
        untransposedType, convResult, rewriter.getI64ArrayAttr(inversePerm));

    // Reshape to match expected output shape (e.g. keepdims=false removes dim)
    if (untransposedShape != llvm::ArrayRef<int64_t>(outputShape))
      result = createReshape(rewriter, loc, result, outputShape);

    rewriter.replaceOp(op, result);
    return mlir::success();
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

    auto inputShape = getShape(input);
    if (inputShape.empty() || inputShape.size() < 2)
      return mlir::failure();

    if (!op.getReduced().hasOneUse())
      return mlir::failure();

    // onnx.Conv only supports float/quantized types, not integer types.
    auto inputElemType =
        mlir::cast<mlir::ShapedType>(input.getType()).getElementType();
    if (!isConvCompatibleElementType(inputElemType))
      return mlir::failure();

    int64_t rank = inputShape.size();

    // Leave rank-5+ ReduceSums for ReduceOpWithAxesInputConverter downstream.
    if (rank > 4)
      return mlir::failure();

    auto axes = getAxesFromReduceSum(axesInput);
    if (axes.size() != 1)
      return mlir::failure();

    // Only handle axis=1 (NCHW channel). Other axes including axis=rank-1
    // (NHWC channel) are handled by ReduceSumSpatialAxisToConvPattern which
    // correctly reshapes the input to place the reduction dim at position 1.
    int64_t axis = normalizeAxis(axes[0], rank);
    if (axis != 1)
      return mlir::failure();

    // Defer rank-4 quantized channel-axis reductions to
    // ReplaceQDQReductionPass.
    if (op.getKeepdims() != 0 && rank == 4 &&
        mlir::isa<mlir::quant::QuantizedType>(inputElemType)) {
      return mlir::failure();
    }

    auto outputShape = getShape(op.getReduced());
    bool validOutputShape =
        (outputShape.size() == inputShape.size() - 1) ||
        (outputShape.size() == inputShape.size() && outputShape[1] == 1);

    if (!validOutputShape)
      return mlir::failure();

    auto outputElemType =
        mlir::cast<mlir::ShapedType>(op.getReduced().getType())
            .getElementType();
    mlir::Value result = convertReductionToConv(rewriter, loc, input,
        /*isMean=*/false, outputShape, outputElemType);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReduceSumSpatialAxisToConvPattern
//===----------------------------------------------------------------------===//
// Handles ReduceSum on any non-batch, non-NCHW-channel axis by transposing
// the reduction axis to the channel position (axis 1), applying channel-wise
// conv reduction, then transposing back.
// Uses permutation [0, axis, axis+1, ..., r-1, 1, ..., axis-1] which
// composes with ConvertToChannelLastPass's transposes to produce only
// size-1 dim permutations (zero-cost reshapes, no data movement).

struct ReduceSumSpatialAxisToConvPattern
    : public OpRewritePattern<ONNXReduceSumOp> {
  using OpRewritePattern<ONNXReduceSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReduceSumOp op, PatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();
    mlir::Value input = op.getData();
    mlir::Value axesInput = op.getAxes();

    auto inputShape = getShape(input);
    if (inputShape.empty() || inputShape.size() < 4)
      return mlir::failure();

    if (!op.getReduced().hasOneUse())
      return mlir::failure();

    int64_t rank = inputShape.size();

    auto axes = getAxesFromReduceSum(axesInput);
    if (axes.size() != 1)
      return mlir::failure();

    int64_t axis = normalizeAxis(axes[0], rank);

    if (axis == 0 || axis == 1)
      return mlir::failure();

    auto inputElemType =
        mlir::cast<mlir::ShapedType>(input.getType()).getElementType();
    if (op.getKeepdims() != 0 && rank == 4 &&
        mlir::isa<mlir::quant::QuantizedType>(inputElemType))
      return mlir::failure();

    int64_t reductionDimSize = inputShape[axis];

    if (reductionDimSize == mlir::ShapedType::kDynamic)
      return mlir::failure();

    auto outputShape = getShape(op.getReduced());
    if (outputShape.empty())
      return mlir::failure();

    auto inputType = mlir::cast<mlir::ShapedType>(input.getType());
    auto outputElemType =
        mlir::cast<mlir::ShapedType>(op.getReduced().getType())
            .getElementType();

    // If rank < 4, reshape to 4D first by appending size-1 dims.
    // This ensures our transpose and ConvertToChannelLastPass's transpose
    // both operate on 4D tensors and compose to a trivial reshape
    // (only moving size-1 dims) rather than producing residual transposes.
    mlir::Value workingInput = input;
    llvm::SmallVector<int64_t> workingShape(
        inputShape.begin(), inputShape.end());
    int64_t workingRank = rank;
    if (rank < 4) {
      workingShape.resize(4, 1);
      workingInput = createReshape(rewriter, loc, input, workingShape);
      workingRank = 4;
    }

    // Build permutation: [0, axis, axis+1, ..., r-1, 1, ..., axis-1]
    llvm::SmallVector<int64_t> perm;
    perm.push_back(0);
    perm.push_back(axis);
    for (int64_t i = axis + 1; i < workingRank; ++i)
      perm.push_back(i);
    for (int64_t i = 1; i < axis; ++i)
      perm.push_back(i);

    llvm::SmallVector<int64_t> transposedInputShape;
    for (auto p : perm)
      transposedInputShape.push_back(workingShape[p]);

    auto transposedInputType = mlir::RankedTensorType::get(
        transposedInputShape, inputType.getElementType());

    mlir::Value transposedInput = rewriter.create<mlir::ONNXTransposeOp>(
        loc, transposedInputType, workingInput, rewriter.getI64ArrayAttr(perm));

    // In transposed domain: reduction is now on axis 1 (channel)
    llvm::SmallVector<int64_t> transposedConvOutputShape;
    for (int64_t i = 0; i < workingRank; ++i)
      transposedConvOutputShape.push_back(i == 1 ? 1 : transposedInputShape[i]);

    mlir::Value convResult =
        convertReductionToConv(rewriter, loc, transposedInput, /*isMean=*/false,
            transposedConvOutputShape, outputElemType);

    // Compute inverse permutation and transpose back
    llvm::SmallVector<int64_t> inversePerm(workingRank);
    for (int64_t i = 0; i < workingRank; ++i)
      inversePerm[perm[i]] = i;

    auto convResultShape = getShape(convResult);
    llvm::SmallVector<int64_t> untransposedShape;
    for (auto p : inversePerm)
      untransposedShape.push_back(convResultShape[p]);

    auto untransposedType =
        mlir::RankedTensorType::get(untransposedShape, outputElemType);

    mlir::Value result = rewriter.create<mlir::ONNXTransposeOp>(loc,
        untransposedType, convResult, rewriter.getI64ArrayAttr(inversePerm));

    // Reshape to match expected output shape (e.g. keepdims=false removes dim)
    if (untransposedShape != llvm::ArrayRef<int64_t>(outputShape))
      result = createReshape(rewriter, loc, result, outputShape);

    rewriter.replaceOp(op, result);
    return mlir::success();
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
      mulConstant = denseAttr.getSplatValue<mlir::APFloat>().convertToFloat();
    } else if (constElemType.isIntOrIndex()) {
      mulConstant = static_cast<float>(
          denseAttr.getSplatValue<mlir::APInt>().getSExtValue());
    } else {
      // Unsupported element type for constant extraction
      return mlir::failure();
    }

    mlir::Value input = reduceMean.getData();
    auto inputShape = getShape(input);
    if (inputShape.empty() || inputShape.size() < 2)
      return mlir::failure();

    // onnx.Conv only supports float/quantized types, not integer types.
    auto inputElemType =
        mlir::cast<mlir::ShapedType>(input.getType()).getElementType();
    if (!isConvCompatibleElementType(inputElemType))
      return mlir::failure();

    int64_t rank = inputShape.size();

    auto axes = getAxesFromReduceMean(reduceMean);
    if (!isChannelWiseReduction(axes, rank))
      return mlir::failure();

    int64_t inputChannel = inputShape[1];

    float expectedCoeff = getMulConstCoefficient(inputChannel);
    float tolerance = 1e-6f;
    if (std::abs(mulConstant - expectedCoeff) > tolerance)
      return mlir::failure();

    auto outputShape = getShape(mulOp.getC());
    auto outputElemType =
        mlir::cast<mlir::ShapedType>(mulOp.getC().getType()).getElementType();

    mlir::Location loc = mulOp.getLoc();
    mlir::Value result = convertReductionToConv(rewriter, loc, input,
        /*isMean=*/true, outputShape, outputElemType);

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

    mlir::Value input = reduceMean.getData();
    auto inputShape = getShape(input);
    if (inputShape.empty() || inputShape.size() < 2)
      return mlir::failure();

    // onnx.Conv only supports float/quantized types, not integer types.
    auto inputElemType =
        mlir::cast<mlir::ShapedType>(input.getType()).getElementType();
    if (!isConvCompatibleElementType(inputElemType))
      return mlir::failure();

    int64_t rank = inputShape.size();

    auto axes = getAxesFromReduceMean(reduceMean);
    if (!isChannelWiseReduction(axes, rank))
      return mlir::failure();

    int64_t inputChannel = inputShape[1];

    if (!isPowerOf2(inputChannel))
      return mlir::failure();

    auto reluOutputShape = getShape(reluOp.getY());
    auto reduceMeanOutputShape = getShape(reduceMean.getReduced());
    auto reduceMeanOutputElemType =
        mlir::cast<mlir::ShapedType>(reduceMean.getReduced().getType())
            .getElementType();

    mlir::Location loc = reluOp.getLoc();
    mlir::Value convResult = convertReductionToConv(rewriter, loc, input,
        /*isMean=*/true, reduceMeanOutputShape, reduceMeanOutputElemType);

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
    patterns.add<ReduceMeanToConvPattern<mlir::ONNXReduceMeanOp>>(context);
    patterns.add<ReduceMeanToConvPattern<mlir::ONNXReduceMeanV13Op>>(context);
    patterns.add<ReduceMeanSpatialAxisToConvPattern<mlir::ONNXReduceMeanOp>>(
        context);
    patterns.add<ReduceMeanSpatialAxisToConvPattern<mlir::ONNXReduceMeanV13Op>>(
        context);
    patterns.add<ReduceSumToConvPattern>(context);
    patterns.add<ReduceSumSpatialAxisToConvPattern>(context);

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

std::unique_ptr<mlir::Pass> createTransferReduceMeanSumToConvPass() {
  return std::make_unique<TransferReduceMeanSumToConvPass>();
}

} // namespace onnx_mlir
