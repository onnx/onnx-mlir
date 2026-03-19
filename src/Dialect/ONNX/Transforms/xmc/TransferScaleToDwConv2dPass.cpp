// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass converts Scale operations to DepthwiseConv2D operations when
// applicable. The transformation:
// - Matches Scale ops that can be represented as depthwise convolutions
// - Inserts reshape operations to convert tensors to 4D format
// - Converts Scale → DepthwiseConv2D with appropriate kernel/stride/pad
// - Handles optional activation operations (relu, leaky-relu, prelu, relu6)
//
// Pattern: input → scale(weights) [→ activation] → output
// Becomes: input → reshape → depthwise-conv2d [→ activation] → reshape → output

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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "transfer-scale-to-dwconv2d"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Create a shape constant for ONNX Reshape
Value createShapeConstant(
    PatternRewriter &rewriter, Location loc, llvm::ArrayRef<int64_t> shape) {
  onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
  return onnxBuilder.constantInt64(shape);
}

/// Check if operation is a supported activation (relu, leaky-relu, prelu,
/// relu6)
bool isActivationOp(Operation *op) {
  if (!op)
    return false;
  return isa<ONNXReluOp, ONNXLeakyReluOp, ONNXPReluOp, ONNXClipOp>(op);
}

/// Get the single-use activation op following a scale, or nullptr
Operation *getFollowingActivation(Operation *op) {
  if (!op->getResult(0).hasOneUse())
    return nullptr;

  Operation *user = *op->getResult(0).getUsers().begin();
  if (isActivationOp(user))
    return user;

  return nullptr;
}

/// Reshape ND tensor to 4D by prepending ones.
/// [d1, d2, ...] → [1, 1, ..., d1, d2, ...]
Value reshapeInputTo4D(PatternRewriter &rewriter, Location loc, Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputShape = inputType.getShape();
  int64_t rank = inputShape.size();

  if (rank == 4) {
    // Already 4D
    return input;
  }

  // Prepend (4 - rank) ones, then append original dimensions.
  llvm::SmallVector<int64_t> newShape;
  for (int64_t i = 0; i < 4 - rank; i++) {
    newShape.push_back(1);
  }
  for (int64_t i = 0; i < rank; i++) {
    newShape.push_back(inputShape[i]);
  }

  auto newType = RankedTensorType::get(newShape, inputType.getElementType());
  auto shapeConst = createShapeConstant(rewriter, loc, newShape);

  return rewriter.create<ONNXReshapeOp>(loc, newType, input, shapeConst)
      .getReshaped();
}

/// Reshape 4D tensor back to original output shape
Value reshapeOutputToOriginal(PatternRewriter &rewriter, Location loc,
    Value input, llvm::ArrayRef<int64_t> outputShape, Type elementType) {
  auto newType = RankedTensorType::get(outputShape, elementType);
  auto shapeConst = createShapeConstant(rewriter, loc, outputShape);

  return rewriter.create<ONNXReshapeOp>(loc, newType, input, shapeConst)
      .getReshaped();
}

/// Reshape 1D weight [C] to 4D for xfe.Conv: [C] → [C, 1, 1, 1] (NHWC
/// depthwise).
Value reshapeWeightTo4D(PatternRewriter &rewriter, Location loc, Value weight) {
  auto weightType = cast<RankedTensorType>(weight.getType());
  auto weightShape = weightType.getShape();
  // Caller ensures weight is 1D (scale from Mul). [C] → [C, 1, 1, 1].
  llvm::SmallVector<int64_t> newShape = {weightShape[0], 1, 1, 1};

  auto newType = RankedTensorType::get(newShape, weightType.getElementType());
  auto shapeConst = createShapeConstant(rewriter, loc, newShape);

  return rewriter.create<ONNXReshapeOp>(loc, newType, weight, shapeConst)
      .getReshaped();
}

/// Transfer onnx_node_name attribute from source to target op
void transferOnnxNodeName(Operation *sourceOp, Operation *targetOp) {
  if (!sourceOp || !targetOp)
    return;
  auto onnxNodeName =
      sourceOp->getAttrOfType<mlir::StringAttr>("onnx_node_name");
  if (onnxNodeName && !onnxNodeName.getValue().empty()) {
    targetOp->setAttr("onnx_node_name", onnxNodeName);
  }
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
// Pattern: Scale to DepthwiseConv2D
//===----------------------------------------------------------------------===//

struct ScaleToDwConv2dPattern : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const override {
    Location loc = mulOp.getLoc();

    // Get operands (Mul is commutative: scale may be getA() or getB())
    Value input;
    Value scale;
    if (mulOp.getA().getDefiningOp<ONNXConstantOp>()) {
      scale = mulOp.getA();
      input = mulOp.getB();
    } else if (mulOp.getB().getDefiningOp<ONNXConstantOp>()) {
      scale = mulOp.getB();
      input = mulOp.getA();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "SKIP: neither operand is a constant\n");
      return failure();
    }

    // Get types
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto scaleType = dyn_cast<RankedTensorType>(scale.getType());

    if (!inputType || !scaleType)
      return failure();

    auto inputShape = inputType.getShape();
    auto scaleShape = scaleType.getShape();
    int64_t inputRank = inputShape.size();
    int64_t scaleRank = scaleShape.size();

    // Scale should be 1D for this pattern
    if (scaleRank != 1)
      return failure();

    // Input should be 1-4D
    if (inputRank < 1 || inputRank > 4)
      return failure();

    // Scale size should match last dimension of input
    if (scaleShape[0] != inputShape[inputRank - 1])
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "Matched Mul: input rank " << inputRank << ", scale size "
               << scaleShape[0] << ", converting to DepthwiseConv2D\n");

    // Check for following activation
    Operation *activationOp = getFollowingActivation(mulOp);

    // Get output shape
    auto outputType = cast<RankedTensorType>(
        activationOp ? activationOp->getResult(0).getType()
                     : mulOp.getResult().getType());
    auto outputShape = outputType.getShape();

    // Reshape input to 4D (prepend 1s if needed).
    Value reshapedInput = reshapeInputTo4D(rewriter, loc, input);

    // Reshape scale to depthwise weight [C, 1, 1, 1] for XFEConv (NHWC)
    Value reshapedWeight = reshapeWeightTo4D(rewriter, loc, scale);

    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value bias = onnxBuilder.none();

    int64_t numChannels =
        scaleShape[0]; // depthwise: groups = number of channels
    auto reshapedInputType = cast<RankedTensorType>(reshapedInput.getType());
    auto convOutputType = RankedTensorType::get(
        reshapedInputType.getShape(), outputType.getElementType());

    auto autoPadAttr = rewriter.getStringAttr("NOTSET");
    auto dilationsAttr = rewriter.getI64ArrayAttr({1, 1});
    auto groupAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, /*isSigned=*/true),
            APInt(64, numChannels, /*isSigned=*/true));
    auto kernelShapeAttr = rewriter.getI64ArrayAttr({1, 1});
    auto padsAttr = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    auto stridesAttr = rewriter.getI64ArrayAttr({1, 1});

    auto dwConvOp =
        rewriter.create<XFEConvOp>(loc, convOutputType, reshapedInput,
            reshapedWeight, bias, rewriter.getStringAttr("NONE"), autoPadAttr,
            dilationsAttr, groupAttr, kernelShapeAttr, padsAttr, stridesAttr);

    transferOnnxNodeName(mulOp, dwConvOp);

    Value result = dwConvOp.getResult();

    // Apply activation if present
    if (activationOp) {
      result = createActivation4D(rewriter, loc, activationOp, result);
    }

    // Reshape output back to original shape
    Value finalOutput = reshapeOutputToOriginal(
        rewriter, loc, result, outputShape, outputType.getElementType());

    // Replace operation
    if (activationOp) {
      rewriter.replaceOp(activationOp, finalOutput);
    } else {
      rewriter.replaceOp(mulOp, finalOutput);
    }

    LLVM_DEBUG(llvm::dbgs() << "SUCCESS: Replaced Mul with xfe.Conv\n");
    return success();
  }
};

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass to transfer Scale operations to DepthwiseConv2D operations.
/// This converts element-wise Scale (implemented as Mul with constant) to
/// DepthwiseConv2D by inserting reshape operations before and after.
struct TransferScaleToDwConv2dPass
    : public PassWrapper<TransferScaleToDwConv2dPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transfer-scale-to-dwconv2d";
  }
  StringRef getDescription() const override {
    return "Convert Scale operations to DepthwiseConv2D operations";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    // Scale → DepthwiseConv2D (with/without activation)
    patterns.add<ScaleToDwConv2dPattern>(ctx);

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

std::unique_ptr<mlir::Pass> createTransferScaleToDwConv2dPass() {
  return std::make_unique<TransferScaleToDwConv2dPass>();
}

} // namespace onnx_mlir
