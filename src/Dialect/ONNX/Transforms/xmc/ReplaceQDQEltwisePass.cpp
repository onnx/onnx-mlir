//===- ReplaceQDQEltwisePass.cpp - Fuse Quantized Eltwise Patterns -------===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
//
// This pass performs advanced fusion patterns on element-wise operations that
// already have quantized types (after quant-types pass). It does NOT perform
// basic Q/DQ fusion (Pattern 1) as that's handled by quant-types pass.
//
// Patterns supported:
// 2. Element-wise with Activation Fusion (12 combinations)
//    - 4 binary ops (Add, Mul, Sub, Div) × 3 activations (ReLU, PReLU,
//    LeakyReLU)
//    - Creates XFE QLinearEltwise ops with fused operation and activation
// 3. BFloat16 with ReLU (4 combinations)
// 4. Post-Quantized ReLU for IPU Strix (2 combinations)
//
// Note: This pass assumes quant-types pass has already run, so operations
// already have !quant.uniform types instead of explicit Q/DQ operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/Support/Debug.h"

#include <cmath>
#include <type_traits>
#include <utility>

#define DEBUG_TYPE "replace-qdq-eltwise"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Convert LeakyReLU alpha to fixed-point representation.
// Returns (M, N) where alpha ≈ M / 2^N for efficient hardware computation.
static std::pair<int64_t, int64_t> getLeakyReluAlphaToPreluFactor(float alpha) {
  int64_t N = 8;
  int64_t M = static_cast<int64_t>(std::llround(std::exp2(N) * alpha));
  return {M, N};
}

// Check if type is quantized
bool isQuantizedType(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    return mlir::isa<mlir::quant::QuantizedType>(tensorType.getElementType());
  }
  return false;
}

// Check if type is BFloat16
bool isBFloat16Type(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    return tensorType.getElementType().isBF16();
  }
  return type.isBF16();
}

// Check if type is INT8 quantized
bool isInt8QuantizedType(Type type) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return false;
  auto uq = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
      tensorType.getElementType());
  if (!uq)
    return false;
  return uq.getStorageType().isInteger(8);
}

// Check if type is float32
bool isFloat32Type(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    return tensorType.getElementType().isF32();
  }
  return type.isF32();
}

//===----------------------------------------------------------------------===//
// Pattern 2: Element-wise with Activation Fusion
// Eltwise_Op(quantized) -> Activation -> Result(quantized)
// Creates: Single onnx.XCOMPILERFusedEltwise op with all fusion information
//===----------------------------------------------------------------------===//

template <typename EltwiseOp>
static llvm::StringRef getEltwiseTypeString() {
  if constexpr (std::is_same_v<EltwiseOp, ONNXAddOp>)
    return "ADD";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXMulOp>)
    return "MUL";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXSubOp>)
    return "SUB";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXDivOp>)
    return "DIV";
  else
    return "";
}

template <typename EltwiseOp, typename ActivationOp>
struct FuseQuantizedEltwiseActivation : public OpRewritePattern<ActivationOp> {
  using OpRewritePattern<ActivationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ActivationOp activationOp, PatternRewriter &rewriter) const override {
    // Match pattern: Eltwise(quantized) -> Activation -> quantized result
    auto eltwiseOp = activationOp.getX().template getDefiningOp<EltwiseOp>();
    if (!eltwiseOp)
      return rewriter.notifyMatchFailure(
          activationOp, "input not from eltwise operation");

    // Check that eltwise inputs and output are quantized
    if (!isQuantizedType(eltwiseOp.getResult().getType()))
      return rewriter.notifyMatchFailure(
          activationOp, "eltwise output not quantized");

    for (auto operand : eltwiseOp->getOperands()) {
      if (!isQuantizedType(operand.getType()))
        return rewriter.notifyMatchFailure(
            activationOp, "eltwise input not quantized");
    }

    // Check activation output is quantized
    if (!isQuantizedType(activationOp.getResult().getType()))
      return rewriter.notifyMatchFailure(
          activationOp, "activation output not quantized");

    LLVM_DEBUG(
        llvm::dbgs()
        << "Fusing quantized eltwise+activation into onnx.XCOMPILERFusedEltwise: "
        << eltwiseOp->getName() << " + " << activationOp->getName() << "\n");

    // Determine operation type
    StringRef opType = getEltwiseTypeString<EltwiseOp>();
    if (opType.empty())
      return rewriter.notifyMatchFailure(
          activationOp, "unsupported eltwise op");

    // Determine nonlinear type and collect attributes
    StringRef nonlinear;
    FloatAttr alphaAttr;
    IntegerAttr preluInAttr, preluShiftAttr;

    if (std::is_same<ActivationOp, ONNXReluOp>::value) {
      nonlinear = "RELU";
    } else if (std::is_same<ActivationOp, ONNXLeakyReluOp>::value) {
      nonlinear = "LEAKYRELU";
      auto leakyReluOp =
          mlir::cast<ONNXLeakyReluOp>(activationOp.getOperation());
      alphaAttr = leakyReluOp.getAlphaAttr();

      // Convert to fixed-point representation (M, N)
      float alpha = alphaAttr.getValue().convertToFloat();
      auto [M, N] = getLeakyReluAlphaToPreluFactor(alpha);
      preluInAttr = rewriter.getI64IntegerAttr(M);
      preluShiftAttr = rewriter.getI64IntegerAttr(N);
    } else {
      // NOTE: XCOMPILERFusedEltwise does not model PReLU slope.
      return rewriter.notifyMatchFailure(
          activationOp, "unsupported activation for fused op");
    }

    auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(eltwiseOp.getLoc(),
        activationOp.getType(), // Result type (quantized)
        eltwiseOp.getOperand(0), eltwiseOp.getOperand(1),
        /*clip_max=*/IntegerAttr(),
        /*clip_min=*/IntegerAttr(),
        /*leakyrelu_alpha=*/alphaAttr,
        /*nonlinear=*/rewriter.getStringAttr(nonlinear),
        /*nonlinear_in_scales=*/FloatAttr(),
        /*nonlinear_in_zeropoints=*/IntegerAttr(),
        /*prelu_in=*/preluInAttr,
        /*prelu_shift=*/preluShiftAttr,
        /*type=*/rewriter.getStringAttr(opType));

    rewriter.replaceOp(activationOp, fusedOp.getResult());

    // Clean up original eltwise
    if (eltwiseOp->use_empty())
      rewriter.eraseOp(eltwiseOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 3: BFloat16 Intermediate Fusion
// Eltwise(f32) -> Activation(f32) -> result
// Where there's an intermediate BF16 conversion that can be optimized
//===----------------------------------------------------------------------===//

template <typename EltwiseOp, typename ActivationOp>
struct FuseBF16IntermediateActivation : public OpRewritePattern<ActivationOp> {
  using OpRewritePattern<ActivationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ActivationOp activationOp, PatternRewriter &rewriter) const override {
    // This pattern looks for: Eltwise -> (implicit BF16 conversion) ->
    // Activation After quant-types, this might appear as type mismatches

    auto eltwiseOp = activationOp.getX().template getDefiningOp<EltwiseOp>();
    if (!eltwiseOp)
      return rewriter.notifyMatchFailure(
          activationOp, "input not from eltwise operation");

    // Check for BF16 intermediate representation
    auto eltwiseResultType = eltwiseOp.getResult().getType();
    auto activationInputType = activationOp.getX().getType();

    // Pattern: eltwise outputs BF16, activation should produce INT8 quantized
    bool hasBF16Intermediate = isBFloat16Type(eltwiseResultType) ||
                               isBFloat16Type(activationInputType);

    if (!hasBF16Intermediate)
      return rewriter.notifyMatchFailure(
          activationOp, "no BF16 intermediate found");

    // Output should be INT8 quantized
    if (!isInt8QuantizedType(activationOp.getResult().getType()))
      return rewriter.notifyMatchFailure(
          activationOp, "activation output not INT8 quantized");

    // Only support Add for Pattern 3
    if (!isa<ONNXAddOp>(eltwiseOp.getOperation()))
      return rewriter.notifyMatchFailure(
          activationOp, "eltwise must be Add for BF16 pattern");

    LLVM_DEBUG(llvm::dbgs()
               << "Fusing BF16 intermediate pattern: " << eltwiseOp->getName()
               << " -> " << activationOp->getName() << "\n");

    // Create optimized path: eltwise with f32, then activation with quantized
    // output
    auto newEltwiseOp = rewriter.create<EltwiseOp>(eltwiseOp.getLoc(),
        // Use f32 intermediate instead of BF16
        RankedTensorType::get(
            mlir::cast<RankedTensorType>(eltwiseResultType).getShape(),
            rewriter.getF32Type()),
        eltwiseOp->getOperands());

    for (auto attr : eltwiseOp->getAttrs()) {
      newEltwiseOp->setAttr(attr.getName(), attr.getValue());
    }

    // Apply activation with quantized output
    // PReLU requires 2 operands (input + slope), others need 1
    Operation *newActivationOp;
    if (std::is_same<ActivationOp, ONNXPReluOp>::value) {
      auto preluOp = mlir::cast<ONNXPReluOp>(activationOp.getOperation());
      newActivationOp = rewriter.create<ONNXPReluOp>(activationOp.getLoc(),
          activationOp.getType(), newEltwiseOp.getResult(), preluOp.getSlope());
    } else {
      newActivationOp = rewriter.create<ActivationOp>(activationOp.getLoc(),
          activationOp.getType(), newEltwiseOp.getResult());
    }

    for (auto attr : activationOp->getAttrs()) {
      newActivationOp->setAttr(attr.getName(), attr.getValue());
    }

    rewriter.replaceOp(activationOp, newActivationOp->getResult(0));

    if (eltwiseOp->use_empty())
      rewriter.eraseOp(eltwiseOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 4: Post-Quantized ReLU for IPU Strix
// Eltwise(f32) -> ReLU(f32) where both should be quantized with same params
// This detects when quantization can be applied more efficiently
//===----------------------------------------------------------------------===//

template <typename EltwiseOp>
struct FusePostQuantizedReLUStrix : public OpRewritePattern<ONNXReluOp> {
  bool enableIPUStrix;

  FusePostQuantizedReLUStrix(MLIRContext *context, bool enableIPUStrix = false)
      : OpRewritePattern<ONNXReluOp>(context), enableIPUStrix(enableIPUStrix) {}

  LogicalResult matchAndRewrite(
      ONNXReluOp reluOp, PatternRewriter &rewriter) const override {
    if (!enableIPUStrix)
      return rewriter.notifyMatchFailure(reluOp, "IPU Strix not enabled");

    // Match: Eltwise -> ReLU where both could be quantized with matching params
    auto eltwiseOp = reluOp.getX().template getDefiningOp<EltwiseOp>();
    if (!eltwiseOp)
      return rewriter.notifyMatchFailure(reluOp, "input not from eltwise");

    // Check if both are currently float but should be quantized
    if (!isFloat32Type(eltwiseOp.getResult().getType()))
      return rewriter.notifyMatchFailure(reluOp, "eltwise not float32");

    if (!isFloat32Type(reluOp.getResult().getType()))
      return rewriter.notifyMatchFailure(reluOp, "relu not float32");

    // For Strix, we want to keep operations in quantized domain
    // This pattern identifies opportunities to maintain quantization
    LLVM_DEBUG(llvm::dbgs() << "Marking post-quantized ReLU pattern for Strix: "
                            << eltwiseOp->getName() << " -> ReLU\n");

    // Add an attribute to signal this should stay quantized through the
    // pipeline
    reluOp->setAttr("strix_keep_quantized", rewriter.getBoolAttr(true));
    eltwiseOp->setAttr("strix_keep_quantized", rewriter.getBoolAttr(true));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ReplaceQDQEltwisePass
    : public PassWrapper<ReplaceQDQEltwisePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceQDQEltwisePass)

  ReplaceQDQEltwisePass() = default;
  ReplaceQDQEltwisePass(const ReplaceQDQEltwisePass &pass) : PassWrapper(pass) {}

  StringRef getArgument() const override { return "replace-qdq-eltwise"; }
  StringRef getDescription() const override {
    return "Fuse quantized eltwise+activation patterns into "
           "onnx.XCOMPILERFusedEltwise";
  }

  Option<bool> enableIPUStrix{*this, "enable-ipu-strix",
      llvm::cl::desc("Enable IPU Strix post-quantized relu annotation"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    auto function = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    //========================================================================
    // Pattern 2: Element-wise with Activation Fusion (12 combinations)
    // 4 binary eltwise ops × 3 activation ops = 12 combinations
    //========================================================================

    // Add + Activations (2 combinations)
    patterns.add<FuseQuantizedEltwiseActivation<ONNXAddOp, ONNXReluOp>>(
        context);
    patterns.add<FuseQuantizedEltwiseActivation<ONNXAddOp, ONNXLeakyReluOp>>(
        context);

    // Mul + Activations (2 combinations)
    patterns.add<FuseQuantizedEltwiseActivation<ONNXMulOp, ONNXReluOp>>(
        context);
    patterns.add<FuseQuantizedEltwiseActivation<ONNXMulOp, ONNXLeakyReluOp>>(
        context);

    // Sub + Activations (2 combinations)
    patterns.add<FuseQuantizedEltwiseActivation<ONNXSubOp, ONNXReluOp>>(
        context);
    patterns.add<FuseQuantizedEltwiseActivation<ONNXSubOp, ONNXLeakyReluOp>>(
        context);

    // Div + Activations (2 combinations)
    patterns.add<FuseQuantizedEltwiseActivation<ONNXDivOp, ONNXReluOp>>(
        context);
    patterns.add<FuseQuantizedEltwiseActivation<ONNXDivOp, ONNXLeakyReluOp>>(
        context);

    // NOTE: Unary ops (Tanh, Sqrt) are NOT supported by XFE QLinearEltwise
    // which requires 2 operands (A and B). Unary ops remain separate.

    //========================================================================
    // Pattern 3: BFloat16 with Activation (4 combinations)
    // 1 eltwise op (Add) × 4 activation ops = 4 combinations
    //========================================================================

    patterns.add<FuseBF16IntermediateActivation<ONNXAddOp, ONNXReluOp>>(
        context);
    patterns.add<FuseBF16IntermediateActivation<ONNXAddOp, ONNXLeakyReluOp>>(
        context);
    // Relu6 would need special handling with Clip

    //========================================================================
    // Pattern 4: Post-Quantized ReLU for IPU Strix (2 combinations)
    // 2 eltwise ops (Add, Mul) × 1 activation (ReLU) = 2 combinations
    //========================================================================

    if (enableIPUStrix) {
      patterns.add<FusePostQuantizedReLUStrix<ONNXAddOp>>(context, true);
      patterns.add<FusePostQuantizedReLUStrix<ONNXMulOp>>(context, true);
    }

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
std::unique_ptr<mlir::Pass> createReplaceQDQEltwisePass() {
  return std::make_unique<ReplaceQDQEltwisePass>();
}
} // namespace onnx_mlir