/*
 *  © Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
 */

//===- FuseConvActivationPass.cpp - Fuse Conv + Activation Patterns ------===//
//
// This pass fuses convolution operations (XFEConv, XFEConvTranspose,
// XCOMPILERDepthwiseConv) with subsequent activation functions by absorbing
// the activation into the convolution op's "activation" attribute.
//
// By the time this pass runs, earlier passes have already lowered raw ONNX
// activation ops into XCOMPILERFusedEltwiseOp:
//   - ReplaceQDQEltwisePass:      Relu       -> FusedEltwise(type="RELU")
//                                 LeakyRelu  -> FusedEltwise(type="LEAKYRELU")
//                                 Clip(0,6)  -> FusedEltwise(type="CLIP")
//                                 Sigmoid    -> FusedEltwise(type="SIGMOID")
//   - ReplaceHsigmoidAndHswishPass: HardSigmoid ->
//   FusedEltwise(type="QLINEARSIGMOID")
//
// The QuantTypes pass (which runs even earlier) converts Q/DQ ops into
// quantized tensor types (e.g., !quant.uniform<i8:f32, ...>). When Q→DQ
// pairs have matching quantization parameters, the canonicalizer folds
// away the resulting scast pair, leaving the conv output flowing directly
// into the activation with quantized types. Therefore, this pass does
// NOT look for explicit quant.scast ops between conv and activation.
// Instead, it checks the quantized tensor types to determine whether an
// original Q→DQ existed:
//   - If conv output and activation output are both quantized with matching
//     scale/zero_point, it is safe to fuse (no requantization).
//   - If quantized types differ, it means a requantization was intended, and
//     we do NOT fuse.
//
// Templates matched:
//
// Template 1: Conv + Activation (inline, after Q→DQ folding)
//   conv -> XCOMPILERFusedEltwiseOp(activation type) -> result
//   Filters:
//     - conv must have single fanout
//     - if both conv output and activation output are quantized, their
//       quantization parameters must match (no requantization)
//
// Template 2: Conv without Activation (default)
//   conv -> result  (activation attribute is "NONE" — no rewrite needed)
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

/// Check if a value has exactly one user (single fanout).
static bool hasSingleUser(Value v) { return v.hasOneUse(); }

/// Check if conv output and activation output have matching quantized types.
/// Fusion is only allowed when there is no requantization between conv and
/// activation (i.e., the scale and zero_point must match).
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

  // If either is not quantized, allow fusion (float path).
  if (!convQuant || !actQuant)
    return true;

  return convQuant.getScale() == actQuant.getScale() &&
         convQuant.getZeroPoint() == actQuant.getZeroPoint() &&
         convQuant.getStorageType() == actQuant.getStorageType();
}

/// Map XCOMPILERFusedEltwiseOp "type" attribute to the conv activation string.
/// Earlier passes convert raw ONNX activation ops into FusedEltwise:
///   Relu        -> type="RELU"
///   LeakyRelu   -> type="LEAKYRELU"
///   Clip(0,6)   -> type="CLIP"             (Relu6)
///   Sigmoid     -> type="SIGMOID"
///   HardSigmoid -> type="QLINEARSIGMOID"
/// Returns the activation string to set on the conv op, or empty if not an
/// activation we fuse.
static StringRef getFusedEltwiseActivationType(
    XCOMPILERFusedEltwiseOp fusedOp) {
  StringRef opType = fusedOp.getType();
  if (opType == "RELU")
    return "RELU";
  if (opType == "LEAKYRELU")
    return "LEAKYRELU";
  if (opType == "CLIP")
    return "RELU6";
  if (opType == "SIGMOID")
    return "SIGMOID";
  if (opType == "QLINEARSIGMOID")
    return "HARDSIGMOID";
  return "";
}

//===----------------------------------------------------------------------===//
// Template 1: Conv + Activation (XCOMPILERFusedEltwise)
//
// ConvOp -> XCOMPILERFusedEltwiseOp(type=RELU|LEAKYRELU|...) -> result
//
// The QuantTypes pass converts Q/DQ into quantized tensor types and the
// canonicalizer folds matching scast pairs away, so by this point the conv
// output flows directly into the activation op with quantized types.
// We check the quantized types to ensure no requantization was intended.
//
// Filters:
//   - conv must have single fanout
//   - conv output and activation output quantized types must match
//===----------------------------------------------------------------------===//

template <typename ConvOp>
struct FuseConvActivation : public OpRewritePattern<XCOMPILERFusedEltwiseOp> {
  using OpRewritePattern<XCOMPILERFusedEltwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XCOMPILERFusedEltwiseOp fusedOp,
      PatternRewriter &rewriter) const override {
    // Check if this FusedEltwise represents a supported activation
    StringRef activationType = getFusedEltwiseActivationType(fusedOp);
    if (activationType.empty())
      return rewriter.notifyMatchFailure(
          fusedOp, "FusedEltwise type is not a fuseable activation");

    // The first operand (A) is the activation input
    Value activationInput = fusedOp.getA();
    auto convOp = activationInput.getDefiningOp<ConvOp>();
    if (!convOp)
      return rewriter.notifyMatchFailure(
          fusedOp, "input A not from target conv op");

    // Conv must already have activation="NONE" (not already fused)
    if (convOp.getActivation() != "NONE")
      return rewriter.notifyMatchFailure(
          fusedOp, "conv already has fused activation");

    // Conv output must have single fanout
    if (!hasSingleUser(convOp.getResult()))
      return rewriter.notifyMatchFailure(fusedOp, "conv has multiple users");

    // Conv output and activation output quantized types must match.
    // If they differ, a requantization exists between conv and activation
    // and we cannot fuse.
    if (!quantTypesMatch(
            convOp.getResult().getType(), fusedOp.getResult().getType()))
      return rewriter.notifyMatchFailure(fusedOp,
          "conv output and activation output quant types differ "
          "(requantization — cannot fuse)");

    LLVM_DEBUG(llvm::dbgs()
               << "FuseConvActivation: fusing " << convOp->getName()
               << " + FusedEltwise(type=" << fusedOp.getType()
               << ") -> activation=" << activationType << "\n");

    // Set the activation attribute on the conv op and transfer any
    // activation-specific attributes from the FusedEltwise op.
    rewriter.modifyOpInPlace(convOp, [&] {
      convOp->setAttr("activation", rewriter.getStringAttr(activationType));

      // Preserve leakyrelu_alpha if present (needed by downstream
      // NormalizeConvActivation pass to determine LEAKYRELU vs PRELU).
      if (auto alphaAttr = fusedOp.getLeakyreluAlphaAttr())
        convOp->setAttr("leakyrelu_alpha", alphaAttr);
    });

    // Replace all uses of the FusedEltwise output with the conv output
    rewriter.replaceOp(fusedOp, convOp.getResult());

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

    //========================================================================
    // Conv + XCOMPILERFusedEltwise (activation) fusion
    //
    // By the time this pass runs:
    //   - QuantTypes pass has embedded Q/DQ into quantized tensor types
    //   - Canonicalizer has folded matching scast pairs
    //   - ReplaceQDQEltwisePass and ReplaceHsigmoidAndHswishPass have
    //     converted all raw ONNX activation ops into FusedEltwise
    //
    // So we match: ConvOp -> XCOMPILERFusedEltwiseOp(activation type)
    // and check quantized types to ensure no requantization.
    //========================================================================

    patterns.add<FuseConvActivation<XFEConvOp>>(context);
    patterns.add<FuseConvActivation<XFEConvTransposeOp>>(context);
    patterns.add<FuseConvActivation<XCOMPILERDepthwiseConvOp>>(context);

    //========================================================================
    // Apply patterns with greedy rewriter
    //========================================================================

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
