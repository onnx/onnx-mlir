// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass converts 1D operations to 2D operations:
// - Conv1D → Conv2D
// - ConvTranspose1D → ConvTranspose2D
// - MaxPool1D → MaxPool2D
// - AveragePool1D → AveragePool2D
// - GlobalMaxPool1D → GlobalMaxPool2D
// - GlobalAveragePool1D → GlobalAveragePool2D
// - DepthwiseConv1D → DepthwiseConv2D
//
// The transformation inserts reshape operations before and after the target
// operation to convert 3D tensors [N, C, L] to 4D tensors [N, C, 1, L] and
// back.

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
#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "transfer-op1d-to-op2d"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Extract int64_t value from IntegerAttr
inline int64_t getIntAttrValue(Attribute attr) {
  return cast<IntegerAttr>(attr).getValue().getSExtValue();
}

/// Extract values from ArrayAttr into vector
llvm::SmallVector<int64_t> getArrayAttrValues(std::optional<ArrayAttr> arr) {
  llvm::SmallVector<int64_t> result;
  if (!arr)
    return result;
  for (auto attr : arr->getValue()) {
    result.push_back(getIntAttrValue(attr));
  }
  return result;
}

/// Extend 1D attribute array to 2D by prepending 1
/// [K] → [1, K] (ONNX format: kH=1, kW=K since conv operates on W dimension)
llvm::SmallVector<int64_t> extend1DAttrTo2D(llvm::ArrayRef<int64_t> arr1d) {
  llvm::SmallVector<int64_t> result;
  result.push_back(1);
  for (auto val : arr1d) {
    result.push_back(val);
  }
  return result;
}

/// Extend 1D pad array to 2D pad array
/// ONNX pads format: [H_begin, W_begin, H_end, W_end]
/// [pad_begin, pad_end] → [0, pad_begin, 0, pad_end]
/// H padding=0 (dummy dimension), W padding=original values (conv dimension)
llvm::SmallVector<int64_t> extend1DPadsTo2D(llvm::ArrayRef<int64_t> pads1d) {
  // pads1d is guaranteed to be size 2 (defaulted if empty)
  return {0, pads1d[0], 0, pads1d[1]};
}

/// Create a shape constant for ONNX Reshape
Value createShapeConstant(
    PatternRewriter &rewriter, Location loc, llvm::ArrayRef<int64_t> shape) {
  onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
  return onnxBuilder.constantInt64(shape);
}

/// Reshape 3D tensor [N, C, L] to 4D tensor for ONNX Conv2D
/// ONNX Conv2D expects input in NCHW format: [N, C, H, W]
/// Normal case: [N, C, L] → [N, C, 1, L] (H=1, W=L)
/// Kernel=1 case: [N, C, L] → [1, N, C, L] (special optimization)
Value reshapeInputTo4D(
    PatternRewriter &rewriter, Location loc, Value input, int64_t kernelSize) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputShape = inputType.getShape();

  llvm::SmallVector<int64_t> newShape;
  if (kernelSize == 1) {
    // Special case for kernel=1: [N, C, L] → [1, N, C, L]
    newShape = {1, inputShape[0], inputShape[1], inputShape[2]};
  } else {
    // Normal case: [N, C, L] → [N, C, 1, L] (NCHW with H=1)
    newShape = {inputShape[0], inputShape[1], 1, inputShape[2]};
  }

  auto newType = RankedTensorType::get(newShape, inputType.getElementType());
  auto shapeConst = createShapeConstant(rewriter, loc, newShape);

  return rewriter.create<ONNXReshapeOp>(loc, newType, input, shapeConst)
      .getReshaped();
}

/// Reshape 4D tensor back to 3D tensor using original output shape
/// Note: input may be unranked, so we pass elementType explicitly
Value reshapeOutputTo3D(PatternRewriter &rewriter, Location loc, Value input,
    llvm::ArrayRef<int64_t> outputShape3D, Type elementType) {
  auto newType = RankedTensorType::get(outputShape3D, elementType);
  auto shapeConst = createShapeConstant(rewriter, loc, outputShape3D);

  return rewriter.create<ONNXReshapeOp>(loc, newType, input, shapeConst)
      .getReshaped();
}

/// Reshape 3D weight [OC, IC, K] to 4D for ONNX Conv2D
/// ONNX Conv2D expects weight in [OC, IC, kH, kW] format
/// [OC, IC, K] → [OC, IC, 1, K] (kH=1, kW=K)
Value reshapeWeightTo4D(PatternRewriter &rewriter, Location loc, Value weight) {
  auto weightType = cast<RankedTensorType>(weight.getType());
  auto weightShape = weightType.getShape();

  // [OC, IC, K] → [OC, IC, 1, K] (ONNX weight format with kH=1)
  llvm::SmallVector<int64_t> newShape = {
      weightShape[0], weightShape[1], 1, weightShape[2]};

  auto newType = RankedTensorType::get(newShape, weightType.getElementType());
  auto shapeConst = createShapeConstant(rewriter, loc, newShape);

  return rewriter.create<ONNXReshapeOp>(loc, newType, weight, shapeConst)
      .getReshaped();
}

/// Check if operation has 1D spatial dimensions (3D tensor)
bool is1DSpatialOp(RankedTensorType inputType) {
  return inputType && inputType.getRank() == 3;
}

/// Get the group attribute from Conv/ConvTranspose op, defaults to 1
template <typename ConvOpType>
int64_t getConvOpGroup(ConvOpType convOp) {
  auto groupAttr = convOp.getGroupAttr();
  return groupAttr ? groupAttr.getValue().getSExtValue() : 1;
}

/// Check if operation is a supported activation (relu, leaky-relu, prelu,
/// relu6)
bool isActivationOp(Operation *op) {
  if (!op)
    return false;
  return isa<ONNXReluOp, ONNXLeakyReluOp, ONNXPReluOp, ONNXClipOp>(op);
}

/// Get the single-use activation op following a conv/pool, or nullptr
Operation *getFollowingActivation(Operation *op) {
  if (!op->getResult(0).hasOneUse())
    return nullptr;

  Operation *user = *op->getResult(0).getUsers().begin();
  if (isActivationOp(user))
    return user;

  return nullptr;
}

/// Create activation op with 4D output type
Value createActivation4D(PatternRewriter &rewriter, Location loc,
    Operation *originalActivation, Value input4D) {
  auto outputType = input4D.getType();

  if (auto reluOp = dyn_cast<ONNXReluOp>(originalActivation)) {
    return rewriter.create<ONNXReluOp>(loc, outputType, input4D).getResult();
  }
  if (auto leakyReluOp = dyn_cast<ONNXLeakyReluOp>(originalActivation)) {
    return rewriter
        .create<ONNXLeakyReluOp>(
            loc, outputType, input4D, leakyReluOp.getAlphaAttr())
        .getResult();
  }
  if (auto preluOp = dyn_cast<ONNXPReluOp>(originalActivation)) {
    Value slope = preluOp.getSlope();
    return rewriter.create<ONNXPReluOp>(loc, outputType, input4D, slope)
        .getResult();
  }
  if (auto clipOp = dyn_cast<ONNXClipOp>(originalActivation)) {
    return rewriter
        .create<ONNXClipOp>(
            loc, outputType, input4D, clipOp.getMin(), clipOp.getMax())
        .getResult();
  }

  llvm_unreachable("Unknown activation type");
}

//===----------------------------------------------------------------------===//
// Pattern: Conv1D/ConvTranspose1D to Conv2D/ConvTranspose2D
// (handles Conv, ConvTranspose, regular and depthwise, with/without activation)
//===----------------------------------------------------------------------===//

template <typename ConvOpType>
struct Conv1dToConv2dPattern : public OpRewritePattern<ConvOpType> {
  using OpRewritePattern<ConvOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ConvOpType convOp, PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();

    // Check if this is a 1D convolution (3D input tensor)
    auto inputType = dyn_cast<RankedTensorType>(convOp.getX().getType());
    if (!is1DSpatialOp(inputType))
      return failure();

    // Verify weight is 3D
    auto weightType = dyn_cast<RankedTensorType>(convOp.getW().getType());
    if (!weightType || weightType.getRank() != 3)
      return failure();

    // Check for following activation
    Operation *activationOp = getFollowingActivation(convOp);

    // Get input/output info
    Value input = convOp.getX();
    Value weight = convOp.getW();
    Value bias = convOp.getB();

    auto outputType = cast<RankedTensorType>(
        activationOp ? activationOp->getResult(0).getType()
                     : convOp.getResult().getType());
    auto outputShape3D = outputType.getShape();

    // Get kernel size for special case handling
    auto kernel1d = getArrayAttrValues(convOp.getKernelShape());
    if (kernel1d.empty())
      kernel1d = {weightType.getShape()[2]};
    int64_t kernelSize = kernel1d[0];

    // Reshape input: [N, C, L] → [N, C, 1, L] or [1, N, C, L] for kernel=1
    Value reshapedInput = reshapeInputTo4D(rewriter, loc, input, kernelSize);

    // Reshape weight: [OC, IC/g, K] or [IC, OC/g, K] → [..., 1, K]
    Value reshapedWeight = reshapeWeightTo4D(rewriter, loc, weight);

    // Get and extend common attributes from 1D to 2D
    auto stride1d = getArrayAttrValues(convOp.getStrides());
    auto dilation1d = getArrayAttrValues(convOp.getDilations());
    auto pads1d = getArrayAttrValues(convOp.getPads());

    if (stride1d.empty())
      stride1d = {1};
    if (dilation1d.empty())
      dilation1d = {1};
    if (pads1d.empty())
      pads1d = {0, 0};

    auto kernel2d = extend1DAttrTo2D(kernel1d);
    auto stride2d = extend1DAttrTo2D(stride1d);
    auto dilation2d = extend1DAttrTo2D(dilation1d);
    auto pads2d = extend1DPadsTo2D(pads1d);

    // Handle ConvTranspose-specific attributes
    llvm::SmallVector<int64_t> outputPadding2d;
    llvm::SmallVector<int64_t> outputShape2d;
    if constexpr (std::is_same_v<ConvOpType, ONNXConvTransposeOp>) {
      auto outputPadding1d = getArrayAttrValues(convOp.getOutputPadding());
      if (!outputPadding1d.empty()) {
        outputPadding2d = extend1DAttrTo2D(outputPadding1d);
      }

      auto outputShape1d = getArrayAttrValues(convOp.getOutputShape());
      if (!outputShape1d.empty()) {
        outputShape2d = extend1DAttrTo2D(outputShape1d);
      }
    }

    // Use unranked type - let ONNX infer Conv2D output shape
    auto conv2dOutputType = UnrankedTensorType::get(inputType.getElementType());

    // Handle bias
    Value conv2dBias = bias;
    if (!bias || isa<NoneType>(bias.getType())) {
      onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
      conv2dBias = onnxBuilder.none();
    }

    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);

    // Create Conv2D or ConvTranspose2D operation
    Value result;
    if constexpr (std::is_same_v<ConvOpType, ONNXConvOp>) {
      // Create Conv2D
      auto conv2dOp = rewriter.create<ONNXConvOp>(loc, conv2dOutputType,
          reshapedInput, reshapedWeight, conv2dBias,
          /*auto_pad=*/convOp.getAutoPadAttr(),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation2d),
          /*group=*/IntegerAttr::get(si64Type, getConvOpGroup(convOp)),
          /*kernel_shape=*/rewriter.getI64ArrayAttr(kernel2d),
          /*pads=*/rewriter.getI64ArrayAttr(pads2d),
          /*strides=*/rewriter.getI64ArrayAttr(stride2d));
      // Infer the output shape
      if (failed(conv2dOp.inferShapes([](Region &) {})))
        return failure();
      result = conv2dOp.getResult();
    } else if constexpr (std::is_same_v<ConvOpType, ONNXConvTransposeOp>) {
      // Create ConvTranspose2D
      auto convTranspose2dOp = rewriter.create<ONNXConvTransposeOp>(loc,
          conv2dOutputType, reshapedInput, reshapedWeight, conv2dBias,
          /*auto_pad=*/convOp.getAutoPadAttr(),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation2d),
          /*group=*/IntegerAttr::get(si64Type, getConvOpGroup(convOp)),
          /*kernel_shape=*/rewriter.getI64ArrayAttr(kernel2d),
          /*output_padding=*/
          outputPadding2d.empty() ? ArrayAttr{}
                                  : rewriter.getI64ArrayAttr(outputPadding2d),
          /*output_shape=*/
          outputShape2d.empty() ? ArrayAttr{}
                                : rewriter.getI64ArrayAttr(outputShape2d),
          /*pads=*/rewriter.getI64ArrayAttr(pads2d),
          /*strides=*/rewriter.getI64ArrayAttr(stride2d));
      // Infer the output shape
      if (failed(convTranspose2dOp.inferShapes([](Region &) {})))
        return failure();
      result = convTranspose2dOp.getResult();
    } else {
      static_assert(std::is_same_v<ConvOpType, ONNXConvOp> ||
                        std::is_same_v<ConvOpType, ONNXConvTransposeOp>,
          "Unsupported convolution operation type");
    }

    // Apply activation if present
    if (activationOp) {
      result = createActivation4D(rewriter, loc, activationOp, result);
    }

    // Reshape output back to 3D
    Value finalOutput = reshapeOutputTo3D(
        rewriter, loc, result, outputShape3D, outputType.getElementType());

    // Replace operation
    if (activationOp) {
      rewriter.replaceOp(activationOp, finalOutput);
    } else {
      rewriter.replaceOp(convOp, finalOutput);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Pool1D to Pool2D (MaxPool, AveragePool)
//===----------------------------------------------------------------------===//

template <typename PoolOp>
struct Pool1dToPool2dPattern : public OpRewritePattern<PoolOp> {
  using OpRewritePattern<PoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      PoolOp poolOp, PatternRewriter &rewriter) const override {
    Location loc = poolOp.getLoc();
    Value input = poolOp.getX();

    // Check if 1D pooling (3D input)
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!is1DSpatialOp(inputType))
      return failure();

    auto inputShape = inputType.getShape();
    auto outputType = cast<RankedTensorType>(poolOp.getResult().getType());
    auto outputShape3D = outputType.getShape();

    // Reshape input: [N, C, L] → [N, C, 1, L]
    llvm::SmallVector<int64_t> input4DShape = {
        inputShape[0], inputShape[1], 1, inputShape[2]};
    auto input4DType =
        RankedTensorType::get(input4DShape, inputType.getElementType());
    auto inputShapeConst = createShapeConstant(rewriter, loc, input4DShape);
    Value reshapedInput =
        rewriter.create<ONNXReshapeOp>(loc, input4DType, input, inputShapeConst)
            .getReshaped();

    // Get and extend common attributes
    auto kernel1d = getArrayAttrValues(poolOp.getKernelShape());
    auto stride1d = getArrayAttrValues(poolOp.getStrides());
    auto pads1d = getArrayAttrValues(poolOp.getPads());

    if (stride1d.empty())
      stride1d = {1};
    if (pads1d.empty())
      pads1d = {0, 0};

    auto kernel2d = extend1DAttrTo2D(kernel1d);
    auto stride2d = extend1DAttrTo2D(stride1d);
    auto pads2d = extend1DPadsTo2D(pads1d);

    // Get and extend dilations (supported by both MaxPool and AveragePool)
    auto dilation1d = getArrayAttrValues(poolOp.getDilations());
    if (dilation1d.empty())
      dilation1d = {1};
    auto dilation2d = extend1DAttrTo2D(dilation1d);

    // Use unranked type - let ONNX infer Pool2D output shape
    auto pool2dOutputType = UnrankedTensorType::get(inputType.getElementType());

    Value pool2dResult;

    // Create Pool2D operation based on type
    if constexpr (std::is_same_v<PoolOp, ONNXMaxPoolSingleOutOp>) {
      // MaxPool: has storage_order
      auto si64Type =
          IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);

      auto maxPool2dOp = rewriter.create<ONNXMaxPoolSingleOutOp>(loc,
          pool2dOutputType, reshapedInput,
          /*auto_pad=*/poolOp.getAutoPadAttr(),
          /*ceil_mode=*/poolOp.getCeilModeAttr(),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation2d),
          /*kernel_shape=*/rewriter.getI64ArrayAttr(kernel2d),
          /*pads=*/rewriter.getI64ArrayAttr(pads2d),
          /*storage_order=*/IntegerAttr::get(si64Type, 0),
          /*strides=*/rewriter.getI64ArrayAttr(stride2d));

      // Infer the output shape
      if (failed(maxPool2dOp.inferShapes([](Region &) {})))
        return failure();
      pool2dResult = maxPool2dOp.getResult();
    } else if constexpr (std::is_same_v<PoolOp, ONNXAveragePoolOp>) {
      // AveragePool: has count_include_pad and dilations, NO storage_order
      auto avgPool2dOp = rewriter.create<ONNXAveragePoolOp>(loc,
          pool2dOutputType, reshapedInput,
          /*auto_pad=*/poolOp.getAutoPadAttr(),
          /*ceil_mode=*/poolOp.getCeilModeAttr(),
          /*count_include_pad=*/poolOp.getCountIncludePadAttr(),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation2d),
          /*kernel_shape=*/rewriter.getI64ArrayAttr(kernel2d),
          /*pads=*/rewriter.getI64ArrayAttr(pads2d),
          /*strides=*/rewriter.getI64ArrayAttr(stride2d));

      // Infer the output shape
      if (failed(avgPool2dOp.inferShapes([](Region &) {})))
        return failure();
      pool2dResult = avgPool2dOp.getResult();
    } else {
      static_assert(std::is_same_v<PoolOp, ONNXMaxPoolSingleOutOp> ||
                        std::is_same_v<PoolOp, ONNXAveragePoolOp>,
          "Unsupported pooling operation type");
    }

    // Reshape output back to 3D
    Value finalOutput = reshapeOutputTo3D(rewriter, loc, pool2dResult,
        outputShape3D, outputType.getElementType());

    rewriter.replaceOp(poolOp, finalOutput);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: GlobalPool1D to GlobalPool2D (GlobalMaxPool, GlobalAveragePool)
//===----------------------------------------------------------------------===//

template <typename GlobalPoolOp>
struct GlobalPool1dToGlobalPool2dPattern
    : public OpRewritePattern<GlobalPoolOp> {
  using OpRewritePattern<GlobalPoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      GlobalPoolOp globalPoolOp, PatternRewriter &rewriter) const override {
    Location loc = globalPoolOp.getLoc();
    Value input = globalPoolOp.getX();

    // Check if 1D pooling (3D input)
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!is1DSpatialOp(inputType))
      return failure();

    auto inputShape = inputType.getShape();
    auto outputType =
        cast<RankedTensorType>(globalPoolOp.getResult().getType());
    auto outputShape3D = outputType.getShape();

    // Reshape input: [N, C, L] → [N, C, 1, L]
    llvm::SmallVector<int64_t> input4DShape = {
        inputShape[0], inputShape[1], 1, inputShape[2]};
    auto input4DType =
        RankedTensorType::get(input4DShape, inputType.getElementType());
    auto inputShapeConst = createShapeConstant(rewriter, loc, input4DShape);
    Value reshapedInput =
        rewriter.create<ONNXReshapeOp>(loc, input4DType, input, inputShapeConst)
            .getReshaped();

    // Create GlobalPool2D - no attributes to extend
    auto pool2dOutputType = UnrankedTensorType::get(inputType.getElementType());
    auto globalPool2dOp =
        rewriter.create<GlobalPoolOp>(loc, pool2dOutputType, reshapedInput);

    // Infer the output shape
    if (failed(globalPool2dOp.inferShapes([](Region &) {})))
      return failure();

    // Reshape output back to 3D
    Value finalOutput = reshapeOutputTo3D(rewriter, loc,
        globalPool2dOp.getResult(), outputShape3D, outputType.getElementType());

    rewriter.replaceOp(globalPoolOp, finalOutput);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass to transfer 1D operations to 2D operations.
/// This converts Conv1D, ConvTranspose1D, MaxPool1D, AveragePool1D,
/// GlobalMaxPool1D, GlobalAveragePool1D, and DepthwiseConv1D to their 2D
/// equivalents by inserting reshape operations before and after.
struct TransferOp1dToOp2dPass
    : public PassWrapper<TransferOp1dToOp2dPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "transfer-op1d-to-op2d"; }
  StringRef getDescription() const override {
    return "Convert 1D operations (Conv1D, ConvTranspose1D, MaxPool1D, "
           "AveragePool1D, GlobalMaxPool1D, GlobalAveragePool1D) to 2D "
           "equivalents";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    // Conv1D → Conv2D (handles regular and depthwise, with/without activation)
    patterns.add<Conv1dToConv2dPattern<ONNXConvOp>>(ctx);

    // ConvTranspose1D → ConvTranspose2D (with/without activation)
    patterns.add<Conv1dToConv2dPattern<ONNXConvTransposeOp>>(ctx);

    // MaxPool1D → MaxPool2D
    patterns.add<Pool1dToPool2dPattern<ONNXMaxPoolSingleOutOp>>(ctx);

    // AveragePool1D → AveragePool2D
    patterns.add<Pool1dToPool2dPattern<ONNXAveragePoolOp>>(ctx);

    // GlobalMaxPool1D → GlobalMaxPool2D
    patterns.add<GlobalPool1dToGlobalPool2dPattern<ONNXGlobalMaxPoolOp>>(ctx);

    // GlobalAveragePool1D → GlobalAveragePool2D
    patterns.add<GlobalPool1dToGlobalPool2dPattern<ONNXGlobalAveragePoolOp>>(
        ctx);

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

std::unique_ptr<mlir::Pass> createTransferOp1dToOp2dPass() {
  return std::make_unique<TransferOp1dToOp2dPass>();
}

} // namespace onnx_mlir
