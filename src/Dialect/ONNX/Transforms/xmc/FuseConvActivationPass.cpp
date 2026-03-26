/*
 *  © Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
 */

//===- FuseConvActivationPass.cpp - Fuse Conv + Activation Patterns ------===//
//
// This pass fuses convolution operations (XFEConv, XFEConvTranspose,
// XCOMPILERDepthwiseConv) with subsequent activation functions by absorbing
// the activation into the convolution op's "activation" attribute.
//
// Activations may appear as:
//   1. XCOMPILERFusedEltwiseOp — quantized activations lowered by
//      ReplaceQDQEltwisePass / ReplaceHsigmoidAndHswishPass
//   2. Raw ONNX activation ops (ONNXReluOp, ONNXLeakyReluOp,
//      ONNXHardSigmoidOp) — non-quantized activations that were not
//      converted to FusedEltwise
//
// A single pattern per conv type walks the conv's unique user and checks
// whether it is a fuseable activation of either kind.
//
// Filters:
//   - conv must have single fanout
//   - if both conv output and activation output are quantized, their
//     quantization parameters must match (no requantization)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fuse-conv-activation"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if conv output and activation output have matching quantized types.
/// Returns true if types match or if either is not quantized (float path).
static bool quantTypesMatch(Type convOutputType, Type activationOutputType) {
  auto convTensor = dyn_cast<RankedTensorType>(convOutputType);
  auto actTensor = dyn_cast<RankedTensorType>(activationOutputType);
  if (!convTensor || !actTensor)
    return true;

  auto convQuant =
      dyn_cast<quant::UniformQuantizedType>(convTensor.getElementType());
  auto actQuant =
      dyn_cast<quant::UniformQuantizedType>(actTensor.getElementType());

  if (!convQuant || !actQuant)
    return true;

  return convQuant.getScale() == actQuant.getScale() &&
         convQuant.getZeroPoint() == actQuant.getZeroPoint() &&
         convQuant.getStorageType() == actQuant.getStorageType();
}

/// Try to identify a fuseable activation from an operation.
/// Handles both XCOMPILERFusedEltwiseOp (quantized path) and raw ONNX
/// activation ops (non-quantized path).
///
/// On success, returns the activation string, the activation's output value,
/// and optionally sets alphaOut for LeakyRelu.
/// On failure, returns empty string.
struct ActivationInfo {
  StringRef activationType;
  Value output;
  float alpha = 0.0f;
  FloatAttr alphaAttr;
};

static ActivationInfo getActivationInfo(Operation *op) {
  ActivationInfo info;

  // --- XCOMPILERFusedEltwiseOp (quantized path) ---
  if (auto fusedOp = dyn_cast<XCOMPILERFusedEltwiseOp>(op)) {
    StringRef opType = fusedOp.getType();
    if (opType == "RELU")
      info.activationType = "RELU";
    else if (opType == "LEAKYRELU")
      info.activationType = "LEAKYRELU";
    else if (opType == "CLIP")
      info.activationType = "RELU6";
    else if (opType == "SIGMOID")
      info.activationType = "SIGMOID";
    else if (opType == "QLINEARSIGMOID")
      info.activationType = "HSIGMOID";
    else
      return info; // empty = not fuseable

    info.output = fusedOp.getResult();
    if (auto attr = fusedOp.getLeakyreluAlphaAttr()) {
      info.alphaAttr = attr;
      info.alpha = attr.getValue().convertToFloat();
    }
    return info;
  }

  // --- Raw ONNX activation ops (non-quantized path) ---
  if (isa<ONNXReluOp>(op)) {
    info.activationType = "RELU";
    info.output = op->getResult(0);
    return info;
  }
  if (auto leakyOp = dyn_cast<ONNXLeakyReluOp>(op)) {
    info.activationType = "LEAKYRELU";
    info.output = leakyOp.getResult();
    if (auto attr = leakyOp.getAlphaAttr()) {
      info.alphaAttr = attr;
      info.alpha = attr.getValue().convertToFloat();
    } else {
      info.alpha = 0.01f; // ONNX default
    }
    return info;
  }
  if (isa<ONNXHardSigmoidOp>(op)) {
    info.activationType = "HSIGMOID";
    info.output = op->getResult(0);
    return info;
  }

  return info; // empty = not fuseable
}

//===----------------------------------------------------------------------===//
// Single pattern: Conv + any Activation
//
// Matches on the ConvOp itself. Inspects its single user to determine if
// it is a fuseable activation (FusedEltwise or raw ONNX op).
//===----------------------------------------------------------------------===//

template <typename ConvOp>
struct FuseConvActivation : public OpRewritePattern<ConvOp> {
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ConvOp convOp, PatternRewriter &rewriter) const override {
    // Conv must already have activation="NONE" (not already fused)
    if (convOp.getActivation() != "NONE")
      return rewriter.notifyMatchFailure(
          convOp, "conv already has fused activation");

    // Conv output must have single fanout
    if (!convOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(convOp, "conv has multiple users");

    // Get the single user and check if it is a fuseable activation
    Operation *userOp = *convOp.getResult().getUsers().begin();
    ActivationInfo actInfo = getActivationInfo(userOp);
    if (actInfo.activationType.empty())
      return rewriter.notifyMatchFailure(
          convOp, "single user is not a fuseable activation");

    // Verify the user's input comes from this conv (not from a different
    // operand position for multi-input ops like FusedEltwise)
    bool inputFromConv = false;
    if (auto fusedOp = dyn_cast<XCOMPILERFusedEltwiseOp>(userOp))
      inputFromConv = (fusedOp.getA() == convOp.getResult());
    else
      inputFromConv = (userOp->getOperand(0) == convOp.getResult());
    if (!inputFromConv)
      return rewriter.notifyMatchFailure(
          convOp, "activation input is not from this conv");

    // Conv output and activation output quantized types must match.
    if (!quantTypesMatch(
            convOp.getResult().getType(), actInfo.output.getType()))
      return rewriter.notifyMatchFailure(convOp,
          "conv output and activation output quant types differ "
          "(requantization — cannot fuse)");

    LLVM_DEBUG(llvm::dbgs()
               << "FuseConvActivation: fusing " << convOp->getName() << " + "
               << userOp->getName()
               << " -> activation=" << actInfo.activationType << "\n");

    // Set the activation attribute and transfer alpha if applicable
    rewriter.modifyOpInPlace(convOp, [&] {
      convOp.setActivationAttr(rewriter.getStringAttr(actInfo.activationType));

      if (actInfo.alphaAttr)
        convOp.setLeakyreluAlphaAttr(actInfo.alphaAttr);
    });

    // Update the conv op's result type if activation changes it
    Type activationOutputType = actInfo.output.getType();
    if (convOp.getResult().getType() != activationOutputType)
      convOp.getResult().setType(activationOutputType);

    // Replace the activation op with the conv output
    rewriter.replaceOp(userOp, convOp.getResult());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct FuseConvActivationPass
    : public PassWrapper<FuseConvActivationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseConvActivationPass)

  FuseConvActivationPass() = default;
  FuseConvActivationPass(const FuseConvActivationPass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const override { return "fuse-conv-activation"; }
  StringRef getDescription() const override {
    return "Fuse Conv + Activation patterns into conv ops with activation "
           "attribute (XFEConv, XFEConvTranspose, XCOMPILERDepthwiseConv)";
  }

  void runOnOperation() override {
    auto function = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    // Single pattern per conv type — inspects the conv's unique user to
    // determine if it is a fuseable activation (FusedEltwise or raw ONNX).
    patterns.add<FuseConvActivation<XFEConvOp>>(context);
    patterns.add<FuseConvActivation<XFEConvTransposeOp>>(context);
    patterns.add<FuseConvActivation<XCOMPILERDepthwiseConvOp>>(context);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = 10;

    if (failed(applyPatternsAndFoldGreedily(
            function, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createFuseConvActivationPass() {
  return std::make_unique<FuseConvActivationPass>();
}
} // namespace onnx_mlir
