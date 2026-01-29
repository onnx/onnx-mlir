// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass converts 1D operations to 2D operations:
// - Conv1D → Conv2D
// - MaxPool1D → MaxPool2D
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
Value createShapeConstant(PatternRewriter &rewriter, Location loc,
    llvm::ArrayRef<int64_t> shape) {
  onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
  return onnxBuilder.constantInt64(shape);
}

/// Reshape 3D tensor [N, C, L] to 4D tensor for ONNX Conv2D
/// ONNX Conv2D expects input in NCHW format: [N, C, H, W]
/// Normal case: [N, C, L] → [N, C, 1, L] (H=1, W=L)
/// Kernel=1 case: [N, C, L] → [1, N, C, L] (special optimization)
Value reshapeInputTo4D(PatternRewriter &rewriter, Location loc, Value input,
    int64_t kernelSize) {
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
  llvm::SmallVector<int64_t> newShape = {weightShape[0], weightShape[1], 1,
      weightShape[2]};

  auto newType = RankedTensorType::get(newShape, weightType.getElementType());
  auto shapeConst = createShapeConstant(rewriter, loc, newShape);

  return rewriter.create<ONNXReshapeOp>(loc, newType, weight, shapeConst)
      .getReshaped();
}

/// Check if operation has 1D spatial dimensions (3D tensor)
bool is1DSpatialOp(RankedTensorType inputType) {
  return inputType && inputType.getRank() == 3;
}

/// Get the group attribute from Conv op, defaults to 1
int64_t getConvGroup(ONNXConvOp convOp) {
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
        .create<ONNXLeakyReluOp>(loc, outputType, input4D,
            leakyReluOp.getAlphaAttr())
        .getResult();
  }
  if (auto preluOp = dyn_cast<ONNXPReluOp>(originalActivation)) {
    Value slope = preluOp.getSlope();
    return rewriter.create<ONNXPReluOp>(loc, outputType, input4D, slope)
        .getResult();
  }
  if (auto clipOp = dyn_cast<ONNXClipOp>(originalActivation)) {
    return rewriter
        .create<ONNXClipOp>(loc, outputType, input4D, clipOp.getMin(),
            clipOp.getMax())
        .getResult();
  }

  llvm_unreachable("Unknown activation type");
}

//===----------------------------------------------------------------------===//
// Conv1D/DepthwiseConv1D to Conv2D Transformation
//===----------------------------------------------------------------------===//

/// Unified transformation for Conv1D → Conv2D (handles both regular and
/// depthwise) For depthwise conv, group == C, which getConvGroup() returns
LogicalResult transformConv1dToConv2d(ONNXConvOp convOp,
    PatternRewriter &rewriter, Operation *activationOp) {
  Location loc = convOp.getLoc();
  Value input = convOp.getX();
  Value weight = convOp.getW();
  Value bias = convOp.getB();

  auto inputType = cast<RankedTensorType>(input.getType());
  auto weightType = cast<RankedTensorType>(weight.getType());

  auto outputType =
      cast<RankedTensorType>(activationOp ? activationOp->getResult(0).getType()
                                          : convOp.getResult().getType());
  auto outputShape3D = outputType.getShape();

  // Get kernel size for special case handling
  auto kernel1d = getArrayAttrValues(convOp.getKernelShape());
  if (kernel1d.empty())
    kernel1d = {weightType.getShape()[2]};
  int64_t kernelSize = kernel1d[0];

  // Reshape input: [N, C, L] → [N, C, 1, L] or [1, N, C, L] for kernel=1
  Value reshapedInput = reshapeInputTo4D(rewriter, loc, input, kernelSize);

  // Reshape weight: [OC, IC/g, K] → [OC, IC/g, 1, K]
  Value reshapedWeight = reshapeWeightTo4D(rewriter, loc, weight);

  // Get and extend attributes from 1D to 2D
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

  // Create Conv2D (group from original op - works for both regular and
  // depthwise)
  auto conv2dOp = rewriter.create<ONNXConvOp>(loc, conv2dOutputType,
      reshapedInput, reshapedWeight, conv2dBias,
      /*auto_pad=*/convOp.getAutoPadAttr(),
      /*dilations=*/rewriter.getI64ArrayAttr(dilation2d),
      /*group=*/IntegerAttr::get(si64Type, getConvGroup(convOp)),
      /*kernel_shape=*/rewriter.getI64ArrayAttr(kernel2d),
      /*pads=*/rewriter.getI64ArrayAttr(pads2d),
      /*strides=*/rewriter.getI64ArrayAttr(stride2d));

  Value result = conv2dOp.getResult();

  if (activationOp) {
    result = createActivation4D(rewriter, loc, activationOp, result);
  }

  Value finalOutput = reshapeOutputTo3D(rewriter, loc, result, outputShape3D,
      outputType.getElementType());

  if (activationOp) {
    rewriter.replaceOp(activationOp, finalOutput);
  } else {
    rewriter.replaceOp(convOp, finalOutput);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pattern: Conv1D to Conv2D (handles both regular and depthwise)
//===----------------------------------------------------------------------===//

struct Conv1dToConv2dPattern : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXConvOp convOp,
      PatternRewriter &rewriter) const override {
    // Check if this is a 1D convolution (3D input tensor)
    auto inputType = dyn_cast<RankedTensorType>(convOp.getX().getType());
    if (!is1DSpatialOp(inputType))
      return failure();

    // Verify weight is 3D
    auto weightType = dyn_cast<RankedTensorType>(convOp.getW().getType());
    if (!weightType || weightType.getRank() != 3)
      return failure();

    // Check for following activation (relu, leaky-relu, prelu, relu6)
    Operation *activationOp = getFollowingActivation(convOp);

    return transformConv1dToConv2d(convOp, rewriter, activationOp);
  }
};

//===----------------------------------------------------------------------===//
// Pattern: MaxPool1D to MaxPool2D
//===----------------------------------------------------------------------===//

struct MaxPool1dToMaxPool2dPattern
    : public OpRewritePattern<ONNXMaxPoolSingleOutOp> {
  using OpRewritePattern<ONNXMaxPoolSingleOutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp maxPoolOp,
      PatternRewriter &rewriter) const override {
    Location loc = maxPoolOp.getLoc();
    Value input = maxPoolOp.getX();

    // Check if 1D pooling (3D input)
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!is1DSpatialOp(inputType))
      return failure();

    auto inputShape = inputType.getShape();
    auto outputType = cast<RankedTensorType>(maxPoolOp.getResult().getType());
    auto outputShape3D = outputType.getShape();

    // Reshape input: [N, C, L] → [N, C, 1, L] (no special kernel=1 case for
    // maxpool)
    llvm::SmallVector<int64_t> input4DShape = {inputShape[0], inputShape[1], 1,
        inputShape[2]};
    auto input4DType =
        RankedTensorType::get(input4DShape, inputType.getElementType());
    auto inputShapeConst = createShapeConstant(rewriter, loc, input4DShape);
    Value reshapedInput =
        rewriter.create<ONNXReshapeOp>(loc, input4DType, input, inputShapeConst)
            .getReshaped();

    // Get and extend attributes
    auto kernel1d = getArrayAttrValues(maxPoolOp.getKernelShape());
    auto stride1d = getArrayAttrValues(maxPoolOp.getStrides());
    auto dilation1d = getArrayAttrValues(maxPoolOp.getDilations());
    auto pads1d = getArrayAttrValues(maxPoolOp.getPads());

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

    // Use unranked type - let ONNX infer MaxPool2D output shape
    auto pool2dOutputType = UnrankedTensorType::get(inputType.getElementType());

    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);

    // Create MaxPool2D
    auto maxPool2dOp = rewriter.create<ONNXMaxPoolSingleOutOp>(loc,
        pool2dOutputType, reshapedInput,
        /*auto_pad=*/maxPoolOp.getAutoPadAttr(),
        /*ceil_mode=*/maxPoolOp.getCeilModeAttr(),
        /*dilations=*/rewriter.getI64ArrayAttr(dilation2d),
        /*kernel_shape=*/rewriter.getI64ArrayAttr(kernel2d),
        /*pads=*/rewriter.getI64ArrayAttr(pads2d),
        /*storage_order=*/IntegerAttr::get(si64Type, 0),
        /*strides=*/rewriter.getI64ArrayAttr(stride2d));

    // Reshape output back to 3D
    Value finalOutput = reshapeOutputTo3D(
        rewriter, loc, maxPool2dOp.getResult(), outputShape3D,
        outputType.getElementType());

    rewriter.replaceOp(maxPoolOp, finalOutput);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass to transfer 1D operations to 2D operations.
/// This converts Conv1D, MaxPool1D, and DepthwiseConv1D to their 2D equivalents
/// by inserting reshape operations before and after.
struct TransferOp1dToOp2dPass
    : public PassWrapper<TransferOp1dToOp2dPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "transfer-op1d-to-op2d"; }
  StringRef getDescription() const override {
    return "Convert 1D operations (Conv1D, MaxPool1D) to 2D equivalents";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    // Conv1D → Conv2D (handles both regular and depthwise, with or without
    // activation)
    patterns.add<Conv1dToConv2dPattern>(ctx);

    // MaxPool1D → MaxPool2D
    patterns.add<MaxPool1dToMaxPool2dPattern>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

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
