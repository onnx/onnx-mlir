// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

using namespace mlir;

namespace {

/// Helper function to create a shape constant for ONNX Reshape
Value createShapeConstant(
    PatternRewriter &rewriter, Location loc, llvm::ArrayRef<int64_t> shape) {
  onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
  return onnxBuilder.constantInt64(shape);
}

/// Check if a value is a constant with specific shape requirements
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

  // Check input dimension <= 4
  if (inputShape.size() > 4)
    return false;

  // Check last dimension matches or weight is scalar
  auto inputLastDim = inputShape[inputShape.size() - 1];
  return (inputLastDim == weightShape[0] || weightShape[0] == 1);
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
    llvm::SmallVector<int64_t, 4> newInputShape;
    llvm::SmallVector<int64_t, 4> newWeightShape;

    // Ensure input is 4D (NHWC)
    if (inputShape.size() < 4) {
      // Pad with 1s at the beginning to make it 4D
      newInputShape.resize(4, 1);
      for (size_t i = 0; i < inputShape.size(); i++) {
        newInputShape[4 - inputShape.size() + i] = inputShape[i];
      }

      // Insert reshape
      auto newInputType =
          RankedTensorType::get(newInputShape, inputType.getElementType());
      auto shapeConst = createShapeConstant(rewriter, loc, newInputShape);
      input =
          rewriter.create<ONNXReshapeOp>(loc, newInputType, input, shapeConst);
    } else {
      newInputShape =
          llvm::SmallVector<int64_t, 4>(inputShape.begin(), inputShape.end());
    }

    // Extract channels from last dimension (NHWC format - Mul broadcasts 1D
    // weights to last dim)
    int64_t C = newInputShape[newInputShape.size() - 1];

    // Create weight tensor for depthwise conv: [M, C/group, kH, kW]
    // For depthwise conv: M = C, C/group = 1, kH = kW = 1
    // So weight shape = [C, 1, 1, 1]
    int64_t inputChannel = C;
    newWeightShape = {inputChannel, 1, 1, 1};

    // Expand weight if needed
    Value newWeight = weight;
    if (weightShape[0] == 1 && inputChannel > 1) {
      // Need to expand weight to match input channels
      auto weightConstOp = weight.getDefiningOp<ONNXConstantOp>();
      if (!weightConstOp)
        return failure();

      // Create new constant with expanded shape
      auto newWeightType =
          RankedTensorType::get(newWeightShape, weightType.getElementType());

      // This would require accessing and expanding the constant data
      // For now, create a reshape
      auto shapeConst = createShapeConstant(rewriter, loc, newWeightShape);
      newWeight = rewriter.create<ONNXReshapeOp>(
          loc, newWeightType, weight, shapeConst);
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

    // Create none value for bias
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value noneBias = onnxBuilder.none();

    // Conv output will be in NCHW format (same as input)
    auto convOutputType =
        RankedTensorType::get(newInputShape, inputType.getElementType());

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

    // Validate pattern
    if (!isValidConstantWeight(weight, input))
      return failure();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();

    // For bias pattern, only support 4D input currently
    if (inputShape.size() != 4)
      return failure();

    // Extract channels from last dimension (NHWC format)
    int64_t C = inputShape[inputShape.size() - 1];

    // Similar to above, create depthwise conv with bias
    auto weightType = cast<RankedTensorType>(weight.getType());

    int64_t inputChannel = C;
    // Weight shape for depthwise conv: [M, C/group, kH, kW] = [inputChannel, 1,
    // 1, 1]
    llvm::SmallVector<int64_t, 4> newWeightShape = {inputChannel, 1, 1, 1};

    // Reshape weight
    auto newWeightType =
        RankedTensorType::get(newWeightShape, weightType.getElementType());
    auto shapeConst = createShapeConstant(rewriter, loc, newWeightShape);
    auto newWeight =
        rewriter.create<ONNXReshapeOp>(loc, newWeightType, weight, shapeConst);

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

    // Create Conv op with bias
    auto convOp = rewriter.create<ONNXConvOp>(loc, convOutputType, input,
        newWeight, /*bias=*/bias,
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

    // Validate pattern
    if (!isValidConstantWeight(weight, input))
      return failure();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto weightType = cast<RankedTensorType>(weight.getType());
    auto inputShape = inputType.getShape();

    // Prepare for depthwise conv
    llvm::SmallVector<int64_t, 4> newInputShape;
    if (inputShape.size() < 4) {
      newInputShape.resize(4, 1);
      for (size_t i = 0; i < inputShape.size(); i++) {
        newInputShape[4 - inputShape.size() + i] = inputShape[i];
      }
      auto newInputType =
          RankedTensorType::get(newInputShape, inputType.getElementType());
      auto shapeConst = createShapeConstant(rewriter, loc, newInputShape);
      input =
          rewriter.create<ONNXReshapeOp>(loc, newInputType, input, shapeConst);
    } else {
      newInputShape =
          llvm::SmallVector<int64_t, 4>(inputShape.begin(), inputShape.end());
    }

    // Extract channels from last dimension (NHWC format)
    int64_t C = newInputShape[newInputShape.size() - 1];

    int64_t inputChannel = C;
    // Weight shape for depthwise conv: [M, C/group, kH, kW] = [inputChannel, 1,
    // 1, 1]
    llvm::SmallVector<int64_t, 4> newWeightShape = {inputChannel, 1, 1, 1};

    auto newWeightType =
        RankedTensorType::get(newWeightShape, weightType.getElementType());
    auto shapeConst = createShapeConstant(rewriter, loc, newWeightShape);
    auto newWeight =
        rewriter.create<ONNXReshapeOp>(loc, newWeightType, weight, shapeConst);

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
        RankedTensorType::get(newInputShape, inputType.getElementType());
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
