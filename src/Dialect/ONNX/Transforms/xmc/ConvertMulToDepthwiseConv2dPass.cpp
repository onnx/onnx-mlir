// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
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

using namespace mlir;

namespace {

/// Skip conversion when the activation input carries a quantized element type.
/// Matches the old xcompiler behavior where TransferMulToDepthwiseConv2dPass
/// returns early when QDQ mode is enabled (qdq::qdq_enabled check).
static bool hasQuantizedInput(Value input) {
  if (auto rtt = dyn_cast<RankedTensorType>(input.getType()))
    return isa<quant::QuantizedType>(rtt.getElementType());
  return false;
}

/// Helper function to create a shape constant for ONNX Reshape
Value createShapeConstant(
    PatternRewriter &rewriter, Location loc, llvm::ArrayRef<int64_t> shape) {
  onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
  return onnxBuilder.constantInt64(shape);
}

/// Expand scalar weight to match input channels and create conv weight
/// directly with shape [C, 1, 1, 1].
/// For quantized types, builds DenseElementsAttr with the storage type
/// and creates ONNXConstantOp with the quantized result type.
Value expandScalarWeight(PatternRewriter &rewriter, Location loc, Value weight,
    int64_t targetChannels, Type elementType) {
  auto weightConstOp = weight.getDefiningOp<ONNXConstantOp>();
  if (!weightConstOp)
    return nullptr;

  auto valueAttr = weightConstOp.getValueAttr();

  if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
    // Get scalar value and replicate for each channel
    llvm::SmallVector<Attribute> values;
    for (int64_t i = 0; i < targetChannels; ++i) {
      values.push_back(*denseAttr.value_begin<Attribute>());
    }

    llvm::SmallVector<int64_t, 4> convWeightShape = {targetChannels, 1, 1, 1};

    // DenseElementsAttr requires a plain integer/float element type.
    // For quantized types, use the storage type for the attribute data.
    Type storageElemType = elementType;
    if (auto qType = dyn_cast<quant::QuantizedType>(elementType))
      storageElemType = qType.getStorageType();

    auto storageType = RankedTensorType::get(convWeightShape, storageElemType);
    auto expandedAttr = DenseElementsAttr::get(storageType, values);

    // Create ONNXConstantOp with the full element type (may be quantized)
    // on the result, while the value attribute uses the storage type.
    auto resultType = RankedTensorType::get(convWeightShape, elementType);
    auto valueNamedAttr = rewriter.getNamedAttr("value", expandedAttr);
    auto newConst =
        rewriter.create<ONNXConstantOp>(loc, resultType, mlir::ValueRange{},
            mlir::ArrayRef<mlir::NamedAttribute>{valueNamedAttr});
    return newConst.getResult();
  }

  return nullptr; // Failed
}

/// Check if bias is per-channel (not per-pixel) for NCHW format.
/// Valid bias shapes: [C], [1, C, 1, 1], or scalar [1]
bool isPerChannelBias(Value bias, int64_t inputChannels) {
  auto biasType = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasType)
    return false;

  auto biasShape = biasType.getShape();

  // Scalar bias [1] - valid
  if (biasShape.size() == 1 && biasShape[0] == 1)
    return true;

  // Per-channel bias [C] - valid
  if (biasShape.size() == 1 && biasShape[0] == inputChannels)
    return true;

  // NCHW broadcast shape [1, C, 1, 1] - valid
  if (biasShape.size() == 4 && biasShape[0] == 1 &&
      biasShape[1] == inputChannels && biasShape[2] == 1 && biasShape[3] == 1)
    return true;

  // Any other shape (e.g., [H, W], [1, C, H, W]) is per-pixel - invalid
  return false;
}

/// Check if a value is a constant with specific shape requirements for
/// conversion to depthwise conv. Only handles 4D NCHW tensors.
bool isValidConstantWeight(Value weight, Value input) {
  auto weightType = dyn_cast<RankedTensorType>(weight.getType());
  auto inputType = dyn_cast<RankedTensorType>(input.getType());

  if (!weightType || !inputType)
    return false;

  auto weightShape = weightType.getShape();
  auto inputShape = inputType.getShape();

  // Check weights is 1D
  if (weightShape.size() != 1)
    return false;

  // Only support exactly 4D input (NCHW format)
  if (inputShape.size() != 4)
    return false;

  // For NCHW format, channels are at index 1
  // Weight size must match channel dimension or be 1 (scalar broadcast)
  int64_t inputChannels = inputShape[1];
  return (inputChannels == weightShape[0] || weightShape[0] == 1);
}

/// Pattern to convert Mul to DepthwiseConv (without bias, without relu)
struct MulToDepthwiseConvPattern : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const override {
    auto loc = mulOp.getLoc();
    Value input;
    Value weight;

    // Try to find which input is the constant weight
    auto lhs = mulOp.getA();
    auto rhs = mulOp.getB();

    if (lhs.getDefiningOp<ONNXConstantOp>()) {
      weight = lhs;
      input = rhs;
    } else if (rhs.getDefiningOp<ONNXConstantOp>()) {
      weight = rhs;
      input = lhs;
    } else {
      return failure(); // Not a constant-input mul
    }

    if (hasQuantizedInput(input))
      return failure();

    // Validate the pattern
    if (!isValidConstantWeight(weight, input))
      return failure();

    // Check if mul has only one use, and if that use is an Add op, skip this
    // pattern (let MulAddToDepthwiseConvPattern handle it)
    if (!mulOp.getResult().hasOneUse())
      return failure();

    auto users = mulOp.getResult().getUsers();
    if (!users.empty()) {
      auto *firstUser = *users.begin();
      if (isa<ONNXAddOp>(firstUser))
        return failure(); // Skip, let MulAddToDepthwiseConvPattern handle it
    }

    auto inputType = cast<RankedTensorType>(input.getType());
    auto weightType = cast<RankedTensorType>(weight.getType());
    auto inputShape = inputType.getShape();
    auto weightShape = weightType.getShape();

    // Prepare shapes for depthwise conv
    llvm::SmallVector<int64_t, 4> newWeightShape;

    // Input is already validated to be 4D NCHW format by isValidConstantWeight
    // Extract channels from index 1 (NCHW format: [N, C, H, W])
    int64_t inputChannel = inputShape[1];

    // Create weight tensor for depthwise conv: [M, C/group, kH, kW]
    // For depthwise conv: M = C, C/group = 1, kH = kW = 1
    // So weight shape = [C, 1, 1, 1]
    newWeightShape = {inputChannel, 1, 1, 1};

    // Expand or reshape weight as needed
    Value newWeight;
    if (weightShape[0] == 1 && inputChannel > 1) {
      // Scalar weight: replicate value for each channel to create [C, 1, 1, 1]
      newWeight = expandScalarWeight(
          rewriter, loc, weight, inputChannel, weightType.getElementType());
      if (!newWeight)
        return failure();
    } else {
      // Weight already has C elements, just reshape to [C, 1, 1, 1]
      auto newWeightType =
          RankedTensorType::get(newWeightShape, weightType.getElementType());
      auto shapeConst = createShapeConstant(rewriter, loc, newWeightShape);
      newWeight = rewriter.create<ONNXReshapeOp>(
          loc, newWeightType, weight, shapeConst);
    }

    // Create DepthwiseConv attributes
    auto kernel = rewriter.getI64ArrayAttr({1, 1});
    auto strides = rewriter.getI64ArrayAttr({1, 1});
    auto dilations = rewriter.getI64ArrayAttr({1, 1});
    auto pads = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    auto group =
        IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
            llvm::APInt(64, inputChannel, /*isSigned=*/true));

    // Create none value for bias
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value noneBias = onnxBuilder.none();

    // Conv output will be in NCHW format (same as input)
    auto convOutputType =
        RankedTensorType::get(inputShape, inputType.getElementType());

    // Create Conv op (DepthwiseConv is Conv with group=channels)
    auto convOp = rewriter.create<ONNXConvOp>(loc, convOutputType, input,
        newWeight, noneBias,
        /*auto_pad=*/rewriter.getStringAttr("NOTSET"),
        /*dilations=*/dilations,
        /*group=*/group,
        /*kernel_shape=*/kernel,
        /*pads=*/pads,
        /*strides=*/strides);

    rewriter.replaceOp(mulOp, convOp.getResult());
    return success();
  }
};

/// Pattern to convert Mul+Add (bias) to DepthwiseConv
struct MulAddToDepthwiseConvPattern : public OpRewritePattern<ONNXAddOp> {
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    auto loc = addOp.getLoc();

    // Check if one input is Mul and other is constant (bias)
    ONNXMulOp mulOp = nullptr;
    Value bias;

    if (auto mul = addOp.getA().getDefiningOp<ONNXMulOp>()) {
      mulOp = mul;
      bias = addOp.getB();
    } else if (auto mul = addOp.getB().getDefiningOp<ONNXMulOp>()) {
      mulOp = mul;
      bias = addOp.getA();
    }

    if (!mulOp || !bias.getDefiningOp<ONNXConstantOp>())
      return failure();

    // Ensure mul has only one use (this add)
    if (!mulOp.getResult().hasOneUse())
      return failure();

    // Get mul's inputs
    Value input;
    Value weight;
    auto lhs = mulOp.getA();
    auto rhs = mulOp.getB();

    if (lhs.getDefiningOp<ONNXConstantOp>()) {
      weight = lhs;
      input = rhs;
    } else if (rhs.getDefiningOp<ONNXConstantOp>()) {
      weight = rhs;
      input = lhs;
    } else {
      return failure();
    }

    if (hasQuantizedInput(input))
      return failure();

    // Validate pattern
    if (!isValidConstantWeight(weight, input))
      return failure();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();

    // For bias pattern, only support 4D input (validated by
    // isValidConstantWeight)
    if (inputShape.size() != 4)
      return failure();

    // Extract channels from index 1 (NCHW format: [N, C, H, W])
    int64_t inputChannel = inputShape[1];

    // Check bias is per-channel broadcast, not per-pixel
    if (!isPerChannelBias(bias, inputChannel))
      return failure();

    // Similar to above, create depthwise conv with bias
    auto weightType = cast<RankedTensorType>(weight.getType());
    auto weightShape = weightType.getShape();
    // Weight shape for depthwise conv: [M, C/group, kH, kW] = [inputChannel, 1,
    // 1, 1]
    llvm::SmallVector<int64_t, 4> newWeightShape = {inputChannel, 1, 1, 1};

    // Expand or reshape weight
    Value newWeight;
    if (weightShape[0] == 1 && inputChannel > 1) {
      newWeight = expandScalarWeight(
          rewriter, loc, weight, inputChannel, weightType.getElementType());
      if (!newWeight)
        return failure();
    } else {
      auto newWeightType =
          RankedTensorType::get(newWeightShape, weightType.getElementType());
      auto shapeConst = createShapeConstant(rewriter, loc, newWeightShape);
      newWeight = rewriter.create<ONNXReshapeOp>(
          loc, newWeightType, weight, shapeConst);
    }

    // Create DepthwiseConv attributes
    auto kernel = rewriter.getI64ArrayAttr({1, 1});
    auto strides = rewriter.getI64ArrayAttr({1, 1});
    auto dilations = rewriter.getI64ArrayAttr({1, 1});
    auto pads = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    auto group =
        IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
            llvm::APInt(64, inputChannel, /*isSigned=*/true));

    // Conv output will be in NCHW format (same as input)
    auto convOutputType =
        RankedTensorType::get(inputShape, inputType.getElementType());

    // Reshape bias constant to [C] to satisfy onnx.Conv verifier:
    // bias dim must equal first dim of weights (inputChannel).
    // isPerChannelBias accepts [C], [1,C,1,1], or [1] - normalize to [C].
    auto biasType = cast<RankedTensorType>(bias.getType());
    auto biasShape = biasType.getShape();
    Value convBias = bias;
    if (!(biasShape.size() == 1 && biasShape[0] == inputChannel)) {
      auto biasConstOp = bias.getDefiningOp<ONNXConstantOp>();
      if (!biasConstOp)
        return failure();
      auto denseAttr = dyn_cast<DenseElementsAttr>(biasConstOp.getValueAttr());
      if (!denseAttr)
        return failure();

      Type storageElemType = biasType.getElementType();
      if (auto qType = dyn_cast<quant::QuantizedType>(storageElemType))
        storageElemType = qType.getStorageType();

      if (biasShape.size() == 1 && biasShape[0] == 1 && inputChannel > 1) {
        // Scalar bias [1]: create splat [C] with the same value
        auto flatStorageType =
            RankedTensorType::get({inputChannel}, storageElemType);
        auto flatAttr = DenseElementsAttr::get(
            flatStorageType, denseAttr.getSplatValue<Attribute>());
        auto flatResultType =
            RankedTensorType::get({inputChannel}, biasType.getElementType());
        convBias =
            rewriter
                .create<ONNXConstantOp>(loc, flatResultType, mlir::ValueRange{},
                    mlir::ArrayRef<mlir::NamedAttribute>{
                        rewriter.getNamedAttr("value", flatAttr)})
                .getResult();
      } else {
        // [1,C,1,1] or other -> reshape dense attr to [C]
        auto flatStorageType =
            RankedTensorType::get({inputChannel}, storageElemType);
        auto flatAttr = denseAttr.reshape(flatStorageType);
        auto flatResultType =
            RankedTensorType::get({inputChannel}, biasType.getElementType());
        convBias =
            rewriter
                .create<ONNXConstantOp>(loc, flatResultType, mlir::ValueRange{},
                    mlir::ArrayRef<mlir::NamedAttribute>{
                        rewriter.getNamedAttr("value", flatAttr)})
                .getResult();
      }
    }

    // Create Conv op with bias
    auto convOp = rewriter.create<ONNXConvOp>(loc, convOutputType, input,
        newWeight, /*bias=*/convBias,
        /*auto_pad=*/rewriter.getStringAttr("NOTSET"),
        /*dilations=*/dilations,
        /*group=*/group,
        /*kernel_shape=*/kernel,
        /*pads=*/pads,
        /*strides=*/strides);

    rewriter.replaceOp(addOp, convOp.getResult());
    return success();
  }
};

/// Pattern to convert Mul+Relu to DepthwiseConv+Relu
struct MulReluToDepthwiseConvPattern : public OpRewritePattern<ONNXReluOp> {
  using OpRewritePattern<ONNXReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReluOp reluOp, PatternRewriter &rewriter) const override {
    auto loc = reluOp.getLoc();

    // Check if input is Mul
    auto mulOp = reluOp.getX().getDefiningOp<ONNXMulOp>();
    if (!mulOp || !mulOp.getResult().hasOneUse())
      return failure();

    // Get mul's inputs
    Value input;
    Value weight;
    auto lhs = mulOp.getA();
    auto rhs = mulOp.getB();

    if (lhs.getDefiningOp<ONNXConstantOp>()) {
      weight = lhs;
      input = rhs;
    } else if (rhs.getDefiningOp<ONNXConstantOp>()) {
      weight = rhs;
      input = lhs;
    } else {
      return failure();
    }

    if (hasQuantizedInput(input))
      return failure();

    // Validate pattern
    if (!isValidConstantWeight(weight, input))
      return failure();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto weightType = cast<RankedTensorType>(weight.getType());
    auto inputShape = inputType.getShape();

    // Only support 4D input (NCHW format) - validated by isValidConstantWeight
    if (inputShape.size() != 4)
      return failure();

    // Extract channels from index 1 (NCHW format: [N, C, H, W])
    int64_t inputChannel = inputShape[1];
    auto weightShape = weightType.getShape();
    // Weight shape for depthwise conv: [M, C/group, kH, kW] = [inputChannel, 1,
    // 1, 1]
    llvm::SmallVector<int64_t, 4> newWeightShape = {inputChannel, 1, 1, 1};

    // Expand or reshape weight
    Value newWeight;
    if (weightShape[0] == 1 && inputChannel > 1) {
      newWeight = expandScalarWeight(
          rewriter, loc, weight, inputChannel, weightType.getElementType());
      if (!newWeight)
        return failure();
    } else {
      auto newWeightType =
          RankedTensorType::get(newWeightShape, weightType.getElementType());
      auto shapeConst = createShapeConstant(rewriter, loc, newWeightShape);
      newWeight = rewriter.create<ONNXReshapeOp>(
          loc, newWeightType, weight, shapeConst);
    }

    // Create DepthwiseConv
    auto kernel = rewriter.getI64ArrayAttr({1, 1});
    auto strides = rewriter.getI64ArrayAttr({1, 1});
    auto dilations = rewriter.getI64ArrayAttr({1, 1});
    auto pads = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    auto group =
        IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
            llvm::APInt(64, inputChannel, /*isSigned=*/true));

    // Create none value for bias
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value noneBias = onnxBuilder.none();

    // Conv output will be in NCHW format (same as input)
    auto convOutputType =
        RankedTensorType::get(inputShape, inputType.getElementType());
    auto convOp = rewriter.create<ONNXConvOp>(loc, convOutputType, input,
        newWeight, noneBias,
        /*auto_pad=*/rewriter.getStringAttr("NOTSET"),
        /*dilations=*/dilations,
        /*group=*/group,
        /*kernel_shape=*/kernel,
        /*pads=*/pads,
        /*strides=*/strides);

    // Create new Relu
    auto newReluOp = rewriter.create<ONNXReluOp>(
        loc, reluOp.getResult().getType(), convOp.getResult());

    rewriter.replaceOp(reluOp, newReluOp.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ConvertMulToDepthwiseConv2dPass
    : public PassWrapper<ConvertMulToDepthwiseConv2dPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "convert-mul-to-depthwise-conv2d";
  }
  StringRef getDescription() const override {
    return "Convert Mul operations to DepthwiseConv2d when applicable";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Add all patterns
    patterns.add<MulToDepthwiseConvPattern>(context);
    patterns.add<MulAddToDepthwiseConvPattern>(context);
    patterns.add<MulReluToDepthwiseConvPattern>(context);

    // Apply patterns greedily
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvertMulToDepthwiseConv2dPass() {
  return std::make_unique<ConvertMulToDepthwiseConv2dPass>();
}

} // namespace onnx_mlir