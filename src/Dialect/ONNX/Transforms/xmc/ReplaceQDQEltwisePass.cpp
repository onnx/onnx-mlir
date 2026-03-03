//===- ReplaceQDQEltwisePass.cpp - Fuse Quantized Eltwise Patterns -------===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
//
// This pass performs fusion patterns on element-wise operations that already
// have quantized types (after quant-types pass). It runs after quant-types, so
// it matches quantized ONNX ops (no explicit QuantizeLinear/DequantizeLinear).
//
// Patterns supported:
// 1. Element-wise (no activation)
//    - Quantized eltwise ops are replaced by onnx.XCOMPILERFusedEltwise with
//      nonlinear="NONE".
//    - Clip is supported via a dedicated pattern that converts constant min/max
//      operands into clip_min/clip_max attributes.
// 2. Element-wise + activation fusion
//    - Quantized binary eltwise ops (Add/Mul/Sub/Div) followed by Relu or
//      LeakyRelu are fused into a single onnx.XCOMPILERFusedEltwise.
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
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
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

// XCOMPILERFusedEltwise requires signed i64 attrs (si64) for prelu_in/shift.
static IntegerAttr getSI64Attr(PatternRewriter &rewriter, int64_t value) {
  MLIRContext *ctx = rewriter.getContext();
  auto si64 =
      IntegerType::get(ctx, 64, IntegerType::SignednessSemantics::Signed);
  return rewriter.getIntegerAttr(si64, value);
}

// Canonicalize activation op types following the xcompiler ReplaceQDQConvPass
// convention. Alpha == 26/256 is the native HW leaky-relu; all other alphas
// (including 0 from plain ReLU) are PRELU with integer mul/shift factors.
// Returns {mappedOpType, leakyrelu_alpha, prelu_in, prelu_shift}.
// mappedOpType is empty when no remapping is needed.
static constexpr float kNativeLeakyReluAlpha = 26.0f / 256.0f;

// ReLU on UINT8 with zero_point=0 is a no-op: unsigned values are already
// non-negative, so clamping at zero has no effect.
static bool isReluNoOp(Operation *op) {
  if (!isa<ONNXReluOp>(op))
    return false;
  auto resultType = op->getResult(0).getType();
  auto tensorType = mlir::dyn_cast<RankedTensorType>(resultType);
  if (!tensorType)
    return false;
  auto uq = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
      tensorType.getElementType());
  if (!uq)
    return false;

  return uq.getStorageType().isUnsignedInteger(8) && uq.getZeroPoint() == 0;
}

static std::tuple<StringRef, FloatAttr, IntegerAttr, IntegerAttr>
computeActivationMapping(Operation *op, PatternRewriter &rewriter) {
  auto mapAlpha = [&](float alpha) {
    auto alphaAttr = rewriter.getFloatAttr(rewriter.getF32Type(), alpha);
    auto [M, N] = getLeakyReluAlphaToPreluFactor(alpha);
    StringRef opType = (alpha == kNativeLeakyReluAlpha) ? "LEAKYRELU" : "PRELU";
    return std::make_tuple(
        opType, alphaAttr, getSI64Attr(rewriter, M), getSI64Attr(rewriter, N));
  };
  if (isa<ONNXReluOp>(op)) {
    if (isReluNoOp(op))
      return {"", FloatAttr(), IntegerAttr(), IntegerAttr()};
    return mapAlpha(0.0f);
  }
  if (auto leakyOp = dyn_cast<ONNXLeakyReluOp>(op)) {
    FloatAttr alphaAttr = leakyOp.getAlphaAttr();
    float alpha = alphaAttr ? alphaAttr.getValue().convertToFloat() : 0.01f;
    return mapAlpha(alpha);
  }
  return {"", FloatAttr(), IntegerAttr(), IntegerAttr()};
}

// True if value is produced by an eltwise op that Pattern 2 fuses with Relu/
// LeakyRelu. Pattern 1 should not fuse standalone Relu/LeakyRelu in that case.
static bool isInputFromPattern2Eltwise(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  return isa<ONNXAddOp, ONNXSubOp, ONNXMulOp, ONNXDivOp, ONNXTanhOp,
      ONNXSqrtOp>(def);
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

// Forward declare for use in Pattern 1 templates.
template <typename EltwiseOp>
static llvm::StringRef getEltwiseTypeString();

// Extract a scalar constant and convert to int64 (for CLIP attrs).
static std::optional<int64_t> getConstScalarI64(Value v) {
  if (!v || isa<NoneType>(v.getType()))
    return std::nullopt;

  auto cst = v.getDefiningOp<ONNXConstantOp>();
  if (!cst)
    return std::nullopt;

  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(cst.getValueAttr());
  if (!elementsAttr)
    return std::nullopt;

  // Splat scalar fast-path.
  if (elementsAttr.isSplat()) {
    Type et = elementsAttr.getElementType();
    if (isa<FloatType>(et)) {
      APFloat apf = elementsAttr.getSplatValue<APFloat>();
      return static_cast<int64_t>(std::llround(apf.convertToDouble()));
    }
    if (auto it = dyn_cast<IntegerType>(et)) {
      APInt api = elementsAttr.getSplatValue<APInt>();
      return static_cast<int64_t>(
          it.isUnsigned() ? api.getZExtValue() : api.getSExtValue());
    }
    return std::nullopt;
  }

  // Non-splat: accept single element only.
  auto shapedTy = dyn_cast<ShapedType>(elementsAttr.getType());
  if (!shapedTy || !shapedTy.hasStaticShape() || shapedTy.getNumElements() != 1)
    return std::nullopt;

  Attribute firstAttr = *elementsAttr.getValues<Attribute>().begin();
  if (auto f = dyn_cast<FloatAttr>(firstAttr))
    return static_cast<int64_t>(std::llround(f.getValueAsDouble()));
  if (auto i = dyn_cast<IntegerAttr>(firstAttr))
    return static_cast<int64_t>(i.getInt());
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Pattern 1: Basic Quantized Element-wise (No Activation)
// Eltwise_Op(quantized) -> Result(quantized)
// Creates: Single onnx.XCOMPILERFusedEltwise with nonlinear="NONE".
// This is the post-quant-types equivalent of the original XCompiler
// dq->eltwise->q template when there is no activation in between.
//===----------------------------------------------------------------------===//

template <typename EltwiseOp>
struct FuseQuantizedEltwiseWithoutActivation
    : public OpRewritePattern<EltwiseOp> {
  using OpRewritePattern<EltwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      EltwiseOp eltwiseOp, PatternRewriter &rewriter) const override {
    // Handle unary and binary eltwise ops.
    Value a = nullptr, b = nullptr;
    bool isUnary = false;
    if (eltwiseOp->getNumOperands() == 1) {
      a = eltwiseOp->getOperand(0);
      isUnary = true;
    } else if (eltwiseOp->getNumOperands() == 2) {
      a = eltwiseOp->getOperand(0);
      b = eltwiseOp->getOperand(1);
    } else {
      return rewriter.notifyMatchFailure(
          eltwiseOp, "eltwise op is not unary/binary");
    }

    // Require quantized operands and result (same as other patterns).
    if (!a || !isQuantizedType(a.getType()) ||
        (!isUnary && !isQuantizedType(b.getType())) ||
        !isQuantizedType(eltwiseOp.getResult().getType()))
      return rewriter.notifyMatchFailure(
          eltwiseOp, "operands/result are not quantized");

    // Standalone Relu/LeakyRelu: do not fuse here if input is from an eltwise
    // that Pattern 2 can fuse with us (Add/Sub/Mul/Div/Tanh/Sqrt+Activation).
    if (isUnary &&
        (isa<ONNXReluOp, ONNXLeakyReluOp>(eltwiseOp.getOperation())) &&
        isInputFromPattern2Eltwise(a))
      return rewriter.notifyMatchFailure(
          eltwiseOp, "eltwise+activation fused by Pattern 2");

    // If this eltwise feeds an activation (ReLU/LeakyReLU/PReLU), don't fuse
    // here. Either Pattern 2 will fuse it (ReLU/LeakyReLU) or we want to keep
    // the original eltwise op intact (PReLU is not modeled by
    // XCOMPILERFusedEltwise).
    for (Operation *user : eltwiseOp.getResult().getUsers())
      if (isa<ONNXReluOp, ONNXLeakyReluOp, ONNXPReluOp>(user))
        return rewriter.notifyMatchFailure(eltwiseOp, "feeds activation op");

    StringRef opType = getEltwiseTypeString<EltwiseOp>();
    if (opType.empty())
      return rewriter.notifyMatchFailure(eltwiseOp, "unsupported eltwise op");

    LLVM_DEBUG(llvm::dbgs()
               << "Fusing quantized eltwise into onnx.XCOMPILERFusedEltwise: "
               << eltwiseOp->getName() << "\n");

    // ReLU on UINT8 with zero_point=0 is a no-op: remove it entirely.
    if (isUnary && isReluNoOp(eltwiseOp.getOperation())) {
      rewriter.replaceOp(eltwiseOp, a);
      return success();
    }
    // Only create IR (e.g. onnx.NoValue) after we know we will rewrite.
    if (isUnary)
      b = rewriter.create<ONNXNoneOp>(eltwiseOp.getLoc()).getResult();

    auto [mappedOpType, leakyAlpha, preluIn, preluShift] =
        computeActivationMapping(eltwiseOp.getOperation(), rewriter);
    if (!mappedOpType.empty())
      opType = mappedOpType;

    // Verifier requires nonlinear=="LEAKYRELU" when prelu attrs are present.
    StringRef nonlinear = "NONE";

    auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(eltwiseOp.getLoc(),
        eltwiseOp.getType(), // result type (quantized)
        a, b,
        /*clip_max=*/IntegerAttr(),
        /*clip_min=*/IntegerAttr(),
        /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
        /*leakyrelu_alpha=*/leakyAlpha,
        /*mul_y=*/FloatAttr(),
        /*nonlinear=*/rewriter.getStringAttr(nonlinear),
        /*nonlinear_in_scales=*/FloatAttr(),
        /*nonlinear_in_zeropoints=*/IntegerAttr(),
        /*prelu_in=*/preluIn,
        /*prelu_shift=*/preluShift,
        /*type=*/rewriter.getStringAttr(opType));

    rewriter.replaceOp(eltwiseOp, fusedOp.getResult());
    return success();
  }
};

// Pattern 1b: Quantized Clip fusion (no activation).
// onnx.Clip(input, min, max) -> onnx.XCOMPILERFusedEltwise(type="CLIP")
// Clip min/max are operands but fused op expects clip_min/clip_max attrs.
struct FuseQuantizedClipWithoutActivation
    : public OpRewritePattern<ONNXClipOp> {
  using OpRewritePattern<ONNXClipOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXClipOp clipOp, PatternRewriter &rewriter) const override {
    if (!isQuantizedType(clipOp.getOutput().getType()) ||
        !isQuantizedType(clipOp.getInput().getType()))
      return rewriter.notifyMatchFailure(clipOp, "clip not quantized");

    // Keep consistent with "no activation" intent.
    for (Operation *user : clipOp.getOutput().getUsers())
      if (isa<ONNXReluOp, ONNXLeakyReluOp, ONNXPReluOp>(user))
        return rewriter.notifyMatchFailure(clipOp, "feeds activation op");

    // Only fuse if min/max are absent or constant scalars, and at least one is
    // present.
    IntegerAttr clipMinAttr, clipMaxAttr;
    if (auto mn = getConstScalarI64(clipOp.getMin()))
      clipMinAttr = getSI64Attr(rewriter, *mn);
    else if (clipOp.getMin() && !isa<NoneType>(clipOp.getMin().getType()))
      return rewriter.notifyMatchFailure(clipOp, "min not constant/none");

    if (auto mx = getConstScalarI64(clipOp.getMax()))
      clipMaxAttr = getSI64Attr(rewriter, *mx);
    else if (clipOp.getMax() && !isa<NoneType>(clipOp.getMax().getType()))
      return rewriter.notifyMatchFailure(clipOp, "max not constant/none");

    if (!clipMinAttr && !clipMaxAttr)
      return rewriter.notifyMatchFailure(
          clipOp, "no constant clip bounds to materialize");

    Value noneB = rewriter.create<ONNXNoneOp>(clipOp.getLoc()).getResult();

    auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(clipOp.getLoc(),
        clipOp.getType(), // result type (quantized)
        clipOp.getInput(), noneB,
        /*clip_max=*/clipMaxAttr,
        /*clip_min=*/clipMinAttr,
        /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
        /*leakyrelu_alpha=*/FloatAttr(),
        /*mul_y=*/FloatAttr(),
        /*nonlinear=*/rewriter.getStringAttr("NONE"),
        /*nonlinear_in_scales=*/FloatAttr(),
        /*nonlinear_in_zeropoints=*/IntegerAttr(),
        /*prelu_in=*/IntegerAttr(),
        /*prelu_shift=*/IntegerAttr(),
        /*type=*/rewriter.getStringAttr("CLIP"));

    rewriter.replaceOp(clipOp, fusedOp.getResult());
    return success();
  }
};

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
  else if constexpr (std::is_same_v<EltwiseOp, ONNXMaxOp>)
    return "MAX";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXMinOp>)
    return "MIN";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXSqrtOp>)
    return "SQRT";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXNegOp>)
    return "NEG";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXTanhOp>)
    return "TANH";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXExpOp>)
    return "EXP";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXEluOp>)
    return "ELU";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXGeluOp>)
    return "GELU";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXSinOp>)
    return "SIN";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXCosOp>)
    return "COS";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXAbsOp>)
    return "ABS";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXSoftplusOp>)
    return "SOFTPLUS";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXEqualOp>)
    return "EQUAL";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXLessOp>)
    return "LESS";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXGreaterOp>)
    return "GREATER";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXLessOrEqualOp>)
    return "LESS_EQUAL";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXGreaterOrEqualOp>)
    return "GREATER_EQUAL";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXModOp>)
    return "MOD";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXReluOp>)
    return "RELU";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXLeakyReluOp>)
    return "LEAKYRELU";
  else if constexpr (std::is_same_v<EltwiseOp, ONNXSigmoidOp>)
    return "SIGMOID";
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

    // Handle unary and binary eltwise ops.
    Value a = nullptr, b = nullptr;
    bool isUnary = false;
    if (eltwiseOp->getNumOperands() == 1) {
      a = eltwiseOp->getOperand(0);
      isUnary = true;
    } else if (eltwiseOp->getNumOperands() == 2) {
      a = eltwiseOp->getOperand(0);
      b = eltwiseOp->getOperand(1);
    } else {
      return rewriter.notifyMatchFailure(
          activationOp, "eltwise op is not unary/binary");
    }

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

    // Verify scale/zp consistency between eltwise output and activation output.
    // If they differ, there's an implicit requantization that would be lost by
    // fusing. (Mirrors xcompiler ReplaceQDQEltwisePass line 764.)
    {
      auto eltwiseResultType =
          mlir::cast<RankedTensorType>(eltwiseOp.getResult().getType());
      auto activationResultType =
          mlir::cast<RankedTensorType>(activationOp.getResult().getType());
      auto eltwiseQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
          eltwiseResultType.getElementType());
      auto activationQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
          activationResultType.getElementType());
      if (eltwiseQType && activationQType &&
          (std::fabs(eltwiseQType.getScale() - activationQType.getScale()) >
                  1e-6 ||
              eltwiseQType.getZeroPoint() != activationQType.getZeroPoint()))
        return rewriter.notifyMatchFailure(activationOp,
            "eltwise and activation have different quantization parameters");
    }

    LLVM_DEBUG(llvm::dbgs() << "Fusing quantized eltwise+activation into "
                               "onnx.XCOMPILERFusedEltwise: "
                            << eltwiseOp->getName() << " + "
                            << activationOp->getName() << "\n");

    // Determine operation type
    StringRef opType = getEltwiseTypeString<EltwiseOp>();
    if (opType.empty())
      return rewriter.notifyMatchFailure(
          activationOp, "unsupported eltwise op");

    // ReLU on UINT8 with zero_point=0 is a no-op. Fuse the eltwise without
    // any activation (nonlinear="NONE"), effectively dropping the ReLU.
    if (isReluNoOp(activationOp.getOperation())) {
      if (isUnary)
        b = rewriter.create<ONNXNoneOp>(eltwiseOp.getLoc()).getResult();
      auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(
          eltwiseOp.getLoc(), activationOp.getType(), a, b,
          /*clip_max=*/IntegerAttr(),
          /*clip_min=*/IntegerAttr(),
          /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
          /*leakyrelu_alpha=*/FloatAttr(),
          /*mul_y=*/FloatAttr(),
          /*nonlinear=*/rewriter.getStringAttr("NONE"),
          /*nonlinear_in_scales=*/FloatAttr(),
          /*nonlinear_in_zeropoints=*/IntegerAttr(),
          /*prelu_in=*/IntegerAttr(),
          /*prelu_shift=*/IntegerAttr(),
          /*type=*/rewriter.getStringAttr(opType));
      rewriter.replaceOp(activationOp, fusedOp.getResult());
      if (eltwiseOp->use_empty())
        rewriter.eraseOp(eltwiseOp);
      return success();
    }

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
      preluInAttr = getSI64Attr(rewriter, M);
      preluShiftAttr = getSI64Attr(rewriter, N);
    } else {
      // NOTE: XCOMPILERFusedEltwise does not model PReLU slope.
      return rewriter.notifyMatchFailure(
          activationOp, "unsupported activation for fused op");
    }

    // Only create IR (e.g. onnx.NoValue) after we know we will rewrite.
    if (isUnary)
      b = rewriter.create<ONNXNoneOp>(eltwiseOp.getLoc()).getResult();

    auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(eltwiseOp.getLoc(),
        activationOp.getType(), // Result type (quantized)
        a, b,
        /*clip_max=*/IntegerAttr(),
        /*clip_min=*/IntegerAttr(),
        /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
        /*leakyrelu_alpha=*/alphaAttr,
        /*mul_y=*/FloatAttr(),
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

    // If already marked, don't keep rewriting forever.
    if (reluOp->hasAttr("strix_keep_quantized"))
      return rewriter.notifyMatchFailure(reluOp, "already marked");

    // Match: Eltwise -> ReLU where both could be quantized with matching params
    auto eltwiseOp = reluOp.getX().template getDefiningOp<EltwiseOp>();
    if (!eltwiseOp)
      return rewriter.notifyMatchFailure(reluOp, "input not from eltwise");

    if (eltwiseOp->hasAttr("strix_keep_quantized"))
      return rewriter.notifyMatchFailure(reluOp, "eltwise already marked");

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
    // pipeline.
    rewriter.modifyOpInPlace(reluOp, [&] {
      reluOp->setAttr("strix_keep_quantized", rewriter.getBoolAttr(true));
    });
    rewriter.modifyOpInPlace(eltwiseOp, [&] {
      eltwiseOp->setAttr("strix_keep_quantized", rewriter.getBoolAttr(true));
    });

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
  ReplaceQDQEltwisePass(const ReplaceQDQEltwisePass &pass)
      : PassWrapper(pass) {}

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
    // Pattern 1: Basic quantized eltwise (no activation).
    //========================================================================

    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXAddOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXMulOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXSubOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXDivOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXMaxOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXMinOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXSqrtOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXNegOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXTanhOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXExpOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXEluOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXGeluOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXSinOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXCosOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXAbsOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXSoftplusOp>>(
        context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXEqualOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXLessOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXGreaterOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXLessOrEqualOp>>(
        context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXGreaterOrEqualOp>>(
        context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXModOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXReluOp>>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXLeakyReluOp>>(
        context);
    patterns.add<FuseQuantizedClipWithoutActivation>(context);
    patterns.add<FuseQuantizedEltwiseWithoutActivation<ONNXSigmoidOp>>(context);

    //========================================================================
    // Pattern 2: Element-wise with Activation Fusion.
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

    // Unary ops + activations.
    patterns.add<FuseQuantizedEltwiseActivation<ONNXTanhOp, ONNXReluOp>>(
        context);
    patterns.add<FuseQuantizedEltwiseActivation<ONNXTanhOp, ONNXLeakyReluOp>>(
        context);
    patterns.add<FuseQuantizedEltwiseActivation<ONNXSqrtOp, ONNXReluOp>>(
        context);
    patterns.add<FuseQuantizedEltwiseActivation<ONNXSqrtOp, ONNXLeakyReluOp>>(
        context);

    //========================================================================
    // Pattern 3: BFloat16 with Activation (4 combinations)
    // 1 eltwise op (Add) × 4 activation ops = 4 combinations
    //========================================================================

    patterns.add<FuseBF16IntermediateActivation<ONNXAddOp, ONNXReluOp>>(
        context);
    patterns.add<FuseBF16IntermediateActivation<ONNXAddOp, ONNXPReluOp>>(
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

    onnx_mlir::ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

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