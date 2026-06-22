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
//      operands into min/max attributes.
// 2. Element-wise + activation fusion
//    - Quantized binary eltwise ops (Add/Mul/Sub/Div) followed by Relu or
//      LeakyRelu are fused into a single onnx.XCOMPILERFusedEltwise.
// 3. BFloat16 with ReLU (4 combinations)
// 4. Post-Quantized ReLU for IPU Strix (2 combinations)
// 5. Replace Expand with Eltwise ADD
//    - Quantized Expand ops are replaced by XCOMPILERFusedEltwise ADD with a
//      zero constant at the target shape. No spatial constraints (matches
//      xcompiler's ReplaceQDQExpandToEltwisePass behavior for quantized ops).
//
// Note: This pass assumes quant-types pass has already run, so operations
// already have !quant.uniform types instead of explicit Q/DQ operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
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

// XCOMPILERFusedEltwise CLAMP min/max are signless i32 (matches XIR/xmodel).
static IntegerAttr getI32Attr(PatternRewriter &rewriter, int64_t value) {
  return rewriter.getI32IntegerAttr(static_cast<int32_t>(value));
}

// Canonicalize activation op types following the xcompiler ReplaceQDQConvPass
// Canonicalize LeakyReLU op type to LEAKYRELU with alpha and prelu factors.
// Returns {mappedOpType, leakyrelu_alpha, prelu_in, prelu_shift}.
// mappedOpType is empty when no remapping is needed.

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
  if (auto leakyOp = dyn_cast<ONNXLeakyReluOp>(op)) {
    FloatAttr alphaAttr = leakyOp.getAlphaAttr();
    float alpha = alphaAttr ? alphaAttr.getValue().convertToFloat() : 0.01f;
    if (!alphaAttr)
      alphaAttr = rewriter.getFloatAttr(rewriter.getF32Type(), alpha);
    auto [M, N] = getLeakyReluAlphaToPreluFactor(alpha);
    return {"LEAKYRELU", alphaAttr, getSI64Attr(rewriter, M),
        getSI64Attr(rewriter, N)};
  }
  return {"", FloatAttr(), IntegerAttr(), IntegerAttr()};
}

// Forward declare for use before definition.
bool isFloat32Type(Type type);

// Extract element type from ranked or unranked tensor.
static Type getElementTypeFromTensor(Type type) {
  if (auto rt = mlir::dyn_cast<RankedTensorType>(type))
    return rt.getElementType();
  if (auto ut = mlir::dyn_cast<UnrankedTensorType>(type))
    return ut.getElementType();
  return {};
}

// Check if two quantized types have matching scale and zero_point.
static bool hasMatchingQuantParams(Type typeA, Type typeB) {
  Type elemA = getElementTypeFromTensor(typeA);
  Type elemB = getElementTypeFromTensor(typeB);
  if (!elemA || !elemB)
    return false;
  auto qA = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemA);
  auto qB = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemB);
  if (!qA || !qB)
    return false;
  return std::fabs(qA.getScale() - qB.getScale()) <= 1e-6 &&
         qA.getZeroPoint() == qB.getZeroPoint();
}

// True if value is produced by an eltwise op that Pattern 2 can fuse with
// Relu/LeakyRelu. Pattern 1 should not fuse standalone Relu/LeakyRelu in that
// case. However, if the quantization parameters between the eltwise output and
// the activation output don't match, Pattern 2 will reject the fusion, so
// Pattern 1 should handle it.
static bool isInputFromPattern2Eltwise(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  // Multi-user eltwise: Pattern 2 won't fuse (it would clone the eltwise and
  // duplicate compute), so do not defer here. Letting this return false lets
  // Pattern 1 emit a standalone activation op (matches the xmodel-flow
  // "golden" form: bare eltwise + standalone Relu).
  if (!def->hasOneUse())
    return false;
  if (!isa<ONNXAddOp, ONNXSubOp, ONNXMulOp, ONNXDivOp, ONNXTanhOp, ONNXSqrtOp>(
          def))
    return false;
  // Check that the activation (user of def's result) has matching quant params.
  // If not, Pattern 2 will reject, so don't defer.
  // When eltwise output is f32 (mixed-precision boundary), Pattern 2 can
  // handle it, so always defer.
  if (isFloat32Type(def->getResult(0).getType()))
    return true;
  for (Operation *user : def->getResult(0).getUsers()) {
    if (isa<ONNXReluOp, ONNXLeakyReluOp>(user)) {
      if (!hasMatchingQuantParams(
              def->getResult(0).getType(), user->getResult(0).getType()))
        return false;
    }
  }
  return true;
}

// Check if type is quantized (handles both ranked and unranked tensors).
bool isQuantizedType(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type))
    return mlir::isa<mlir::quant::QuantizedType>(tensorType.getElementType());
  if (auto tensorType = mlir::dyn_cast<UnrankedTensorType>(type))
    return mlir::isa<mlir::quant::QuantizedType>(tensorType.getElementType());
  return false;
}

// Recover a ranked result type for an eltwise op whose result became unranked
// due to upstream shape inference issues with mismatched quantized element
// types. For binary ops (e.g. ADD with broadcast), the result shape is the
// broadcast of both inputs, not one operand's shape.
static Type recoverRankedResultType(Type resultType, Value a, Value b) {
  if (mlir::isa<RankedTensorType>(resultType))
    return resultType;
  auto unrankedType = mlir::dyn_cast<UnrankedTensorType>(resultType);
  if (!unrankedType)
    return resultType;
  auto aType = mlir::dyn_cast<RankedTensorType>(a.getType());
  auto bType = b ? mlir::dyn_cast<RankedTensorType>(b.getType()) : nullptr;
  if (aType && bType) {
    // Binary op: result shape is the broadcast of a and b.
    llvm::SmallVector<int64_t> broadcastedShape;
    if (OpTrait::util::getBroadcastedShape(
            aType.getShape(), bType.getShape(), broadcastedShape))
      return RankedTensorType::get(
          broadcastedShape, unrankedType.getElementType());
    // Incompatible shapes: fall through to single-operand fallback.
  }
  if (aType)
    return RankedTensorType::get(
        aType.getShape(), unrankedType.getElementType());
  if (bType)
    return RankedTensorType::get(
        bType.getShape(), unrankedType.getElementType());
  return resultType;
}

static void maybeWidenNarrowConstOperand(PatternRewriter &rewriter,
    Location loc, StringRef opType, Value &a, Value &b) {
  if (opType != "ADD" && opType != "MUL" && opType != "SUB" && opType != "DIV")
    return;
  if (!a || !b || mlir::isa<NoneType>(b.getType()))
    return;

  auto aTy = mlir::dyn_cast<RankedTensorType>(a.getType());
  auto bTy = mlir::dyn_cast<RankedTensorType>(b.getType());
  if (!aTy || !bTy)
    return;

  auto aQ =
      mlir::dyn_cast<mlir::quant::UniformQuantizedType>(aTy.getElementType());
  auto bQ =
      mlir::dyn_cast<mlir::quant::UniformQuantizedType>(bTy.getElementType());
  if (!aQ || !bQ)
    return;

  auto aStor = mlir::dyn_cast<IntegerType>(aQ.getStorageType());
  auto bStor = mlir::dyn_cast<IntegerType>(bQ.getStorageType());
  if (!aStor || !bStor)
    return;

  unsigned aW = aStor.getWidth();
  unsigned bW = bStor.getWidth();

  bool narrowIsA;
  Value narrowSide;
  mlir::quant::UniformQuantizedType narrowQ;
  RankedTensorType narrowTy;

  if (aW != bW) {
    // Width mismatch: widen the narrow 8-bit operand to its 16-bit sibling.
    narrowIsA = aW < bW;
    narrowSide = narrowIsA ? a : b;
    narrowQ = narrowIsA ? aQ : bQ;
    narrowTy = narrowIsA ? aTy : bTy;
    unsigned narrowW = narrowIsA ? aW : bW;
    unsigned wideW = narrowIsA ? bW : aW;
    if (narrowW != 8 || wideW != 16)
      return;
  } else {
    // Same-width mixed signedness: promote INT8 const to INT16 (golden).
    if (aW != 8)
      return;
    if (aQ.isSigned() && !bQ.isSigned() && a.getDefiningOp<ONNXConstantOp>()) {
      narrowIsA = true;
      narrowSide = a;
      narrowQ = aQ;
      narrowTy = aTy;
    } else if (bQ.isSigned() && !aQ.isSigned() &&
               b.getDefiningOp<ONNXConstantOp>()) {
      narrowIsA = false;
      narrowSide = b;
      narrowQ = bQ;
      narrowTy = bTy;
    } else {
      return;
    }
  }

  auto constOp = narrowSide.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return;

  auto valueAttr =
      mlir::dyn_cast_or_null<DenseElementsAttr>(constOp.getValueAttr());
  if (!valueAttr || !valueAttr.getElementType().isIntOrIndex())
    return;

  bool isSigned = narrowQ.isSigned();
  auto wideStorTy = rewriter.getIntegerType(16);

  auto wideQ = mlir::quant::UniformQuantizedType::get(narrowQ.getFlags(),
      wideStorTy, narrowQ.getExpressedType(), narrowQ.getScale(),
      narrowQ.getZeroPoint(),
      mlir::quant::UniformQuantizedType::getDefaultMinimumForInteger(
          isSigned, 16),
      mlir::quant::UniformQuantizedType::getDefaultMaximumForInteger(
          isSigned, 16));

  auto wideTensorTy = RankedTensorType::get(narrowTy.getShape(), wideQ);

  // Reuse an existing widened constant if a previous pattern match already
  // created one for the same narrow constant (avoids duplicating memory).
  // Match requires identical type AND identical dense values.
  Value wideResult;
  for (Operation &sibling : constOp->getBlock()->getOperations()) {
    auto candidate = mlir::dyn_cast<ONNXConstantOp>(&sibling);
    if (!candidate || candidate == constOp)
      continue;
    if (candidate.getResult().getType() != wideTensorTy)
      continue;
    auto candidateVal =
        mlir::dyn_cast_or_null<DenseElementsAttr>(candidate.getValueAttr());
    if (candidateVal &&
        candidateVal.getNumElements() == valueAttr.getNumElements()) {
      bool valuesMatch = true;
      auto candIt = candidateVal.getValues<llvm::APInt>().begin();
      for (llvm::APInt v : valueAttr.getValues<llvm::APInt>()) {
        llvm::APInt expected = isSigned ? v.sext(16) : v.zext(16);
        if (*candIt != expected) {
          valuesMatch = false;
          break;
        }
        ++candIt;
      }
      if (valuesMatch) {
        wideResult = candidate.getResult();
        break;
      }
    }
  }

  if (!wideResult) {
    auto wideStorageTensorTy =
        RankedTensorType::get(narrowTy.getShape(), wideStorTy);
    llvm::SmallVector<llvm::APInt, 16> widened;
    widened.reserve(valueAttr.getNumElements());
    for (llvm::APInt v : valueAttr.getValues<llvm::APInt>())
      widened.push_back(isSigned ? v.sext(16) : v.zext(16));
    auto wideDense = DenseElementsAttr::get(
        wideStorageTensorTy, llvm::ArrayRef<llvm::APInt>(widened));
    auto wideValueAttr = rewriter.getNamedAttr("value", wideDense);
    auto wideConstOp = rewriter.create<ONNXConstantOp>(loc, wideTensorTy,
        ValueRange{}, llvm::ArrayRef<NamedAttribute>{wideValueAttr});
    wideResult = wideConstOp.getResult();
  }

  if (narrowIsA)
    a = wideResult;
  else
    b = wideResult;

  if (constOp->use_empty())
    rewriter.eraseOp(constOp);
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

    // Only create IR (e.g. onnx.NoValue) after we know we will rewrite.
    if (isUnary)
      b = rewriter.create<ONNXNoneOp>(eltwiseOp.getLoc()).getResult();

    auto [mappedOpType, leakyAlpha, preluIn, preluShift] =
        computeActivationMapping(eltwiseOp.getOperation(), rewriter);
    if (!mappedOpType.empty())
      opType = mappedOpType;

    StringRef nonlinear = "NONE";

    Type resultType = recoverRankedResultType(eltwiseOp.getType(), a, b);

    maybeWidenNarrowConstOperand(rewriter, eltwiseOp.getLoc(), opType, a, b);

    // Carry GELU's approximation mode ("none"/"tanh") onto the fused op so the
    // kernel can pick the matching GELU formulation. Null for non-GELU types.
    StringAttr approximateAttr;
    if constexpr (std::is_same_v<EltwiseOp, ONNXGeluOp>)
      approximateAttr = rewriter.getStringAttr(eltwiseOp.getApproximate());

    auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(eltwiseOp.getLoc(),
        resultType, a, b,
        /*approximate=*/approximateAttr,
        /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
        /*leakyrelu_alpha=*/leakyAlpha,
        /*max=*/IntegerAttr(),
        /*min=*/IntegerAttr(),
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
// onnx.Clip(input, min, max) -> onnx.XCOMPILERFusedEltwise(type="CLAMP")
// Clip min/max are operands but fused op expects min/max attrs.
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
      clipMinAttr = getI32Attr(rewriter, *mn);
    else if (clipOp.getMin() && !isa<NoneType>(clipOp.getMin().getType()))
      return rewriter.notifyMatchFailure(clipOp, "min not constant/none");

    if (auto mx = getConstScalarI64(clipOp.getMax()))
      clipMaxAttr = getI32Attr(rewriter, *mx);
    else if (clipOp.getMax() && !isa<NoneType>(clipOp.getMax().getType()))
      return rewriter.notifyMatchFailure(clipOp, "max not constant/none");

    if (!clipMinAttr && !clipMaxAttr)
      return rewriter.notifyMatchFailure(
          clipOp, "no constant clip bounds to materialize");

    Value noneB = rewriter.create<ONNXNoneOp>(clipOp.getLoc()).getResult();

    auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(clipOp.getLoc(),
        clipOp.getType(), // result type (quantized)
        clipOp.getInput(), noneB,
        /*approximate=*/StringAttr(),
        /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
        /*leakyrelu_alpha=*/FloatAttr(),
        /*max=*/clipMaxAttr,
        /*min=*/clipMinAttr,
        /*mul_y=*/FloatAttr(),
        /*nonlinear=*/rewriter.getStringAttr("NONE"),
        /*nonlinear_in_scales=*/FloatAttr(),
        /*nonlinear_in_zeropoints=*/IntegerAttr(),
        /*prelu_in=*/IntegerAttr(),
        /*prelu_shift=*/IntegerAttr(),
        /*type=*/rewriter.getStringAttr("CLAMP"));

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

    // Multi-user eltwise: refuse to fuse so we do not clone the eltwise into
    // the activation slot (which duplicates compute for every non-activation
    // consumer of the eltwise). Pattern 1 then emits the activation as a
    // standalone XCOMPILERFusedEltwise, matching the xmodel-flow "golden"
    // form (e.g. ADD + standalone RELU for an Add whose result feeds both a
    // Relu and another consumer such as Concat).
    if (!eltwiseOp->hasOneUse())
      return rewriter.notifyMatchFailure(activationOp,
          "eltwise has multiple users; emit standalone activation instead "
          "of cloning the eltwise");

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

    // Check that eltwise inputs are quantized
    for (auto operand : eltwiseOp->getOperands()) {
      if (!isQuantizedType(operand.getType()))
        return rewriter.notifyMatchFailure(
            activationOp, "eltwise input not quantized");
    }

    // Eltwise output can be quantized or f32 (mixed-precision boundary).
    // When f32, the activation bridges back to quantized.
    bool eltwiseOutputIsFloat = isFloat32Type(eltwiseOp.getResult().getType());
    if (!eltwiseOutputIsFloat &&
        !isQuantizedType(eltwiseOp.getResult().getType()))
      return rewriter.notifyMatchFailure(
          activationOp, "eltwise output not quantized or f32");

    // Check activation output is quantized
    if (!isQuantizedType(activationOp.getResult().getType()))
      return rewriter.notifyMatchFailure(
          activationOp, "activation output not quantized");

    // When eltwise output is quantized, verify scale/zp consistency with
    // activation output. If they differ, there's an implicit requantization
    // that would be lost by fusing. When eltwise output is f32, skip this
    // check since there are no quant params to compare.
    if (!eltwiseOutputIsFloat &&
        !hasMatchingQuantParams(eltwiseOp.getResult().getType(),
            activationOp.getResult().getType()))
      return rewriter.notifyMatchFailure(activationOp,
          "eltwise and activation have different quantization parameters");

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
      maybeWidenNarrowConstOperand(rewriter, eltwiseOp.getLoc(), opType, a, b);
      auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(
          eltwiseOp.getLoc(), activationOp.getType(), a, b,
          /*approximate=*/StringAttr(),
          /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
          /*leakyrelu_alpha=*/FloatAttr(),
          /*max=*/IntegerAttr(),
          /*min=*/IntegerAttr(),
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

    maybeWidenNarrowConstOperand(rewriter, eltwiseOp.getLoc(), opType, a, b);

    auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(eltwiseOp.getLoc(),
        activationOp.getType(), // Result type (quantized)
        a, b,
        /*approximate=*/StringAttr(),
        /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
        /*leakyrelu_alpha=*/alphaAttr,
        /*max=*/IntegerAttr(),
        /*min=*/IntegerAttr(),
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
// Pattern 5: Replace Expand with Eltwise ADD
// Quantized Expand(input, shape) -> XCOMPILERFusedEltwise(input, zeros, "ADD")
// Mirrors xcompiler's ReplaceQDQExpandToEltwisePass: no spatial constraints
// since we already gate on quantized types.
//===----------------------------------------------------------------------===//

struct ReplaceExpandWithEltwise : public OpRewritePattern<ONNXExpandOp> {
  using OpRewritePattern<ONNXExpandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXExpandOp expandOp, PatternRewriter &rewriter) const override {
    Value input = expandOp.getInput();
    Type resultType = expandOp.getOutput().getType();

    if (!isQuantizedType(input.getType()) || !isQuantizedType(resultType))
      return rewriter.notifyMatchFailure(expandOp, "not quantized");

    auto inputRankedType = mlir::dyn_cast<RankedTensorType>(input.getType());
    auto outputRankedType = mlir::dyn_cast<RankedTensorType>(resultType);
    if (!inputRankedType || !outputRankedType)
      return rewriter.notifyMatchFailure(expandOp, "not ranked tensors");

    auto inputShape = inputRankedType.getShape();
    if (inputShape.size() == 4 && inputShape[2] != 1)
      return rewriter.notifyMatchFailure(
          expandOp, "4D input requires dim[2] == 1");

    LLVM_DEBUG(llvm::dbgs() << "Replacing quantized Expand with eltwise ADD: "
                            << expandOp->getName() << "\n");

    auto outputShape = outputRankedType.getShape();
    Type elemType = outputRankedType.getElementType();

    // Build a zero constant with the output shape and storage type.
    Type storageType = elemType;
    if (auto qType =
            mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType))
      storageType = qType.getStorageType();

    auto zeroAttr =
        DenseElementsAttr::get(RankedTensorType::get(outputShape, storageType),
            rewriter.getZeroAttr(storageType));

    auto valueNamedAttr = rewriter.getNamedAttr("value", zeroAttr);
    auto zeroConst = rewriter.create<ONNXConstantOp>(expandOp.getLoc(),
        outputRankedType, mlir::ValueRange{},
        mlir::ArrayRef<mlir::NamedAttribute>{valueNamedAttr});

    auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(expandOp.getLoc(),
        resultType, input, zeroConst.getResult(),
        /*approximate=*/StringAttr(),
        /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
        /*leakyrelu_alpha=*/FloatAttr(),
        /*max=*/IntegerAttr(),
        /*min=*/IntegerAttr(),
        /*mul_y=*/FloatAttr(),
        /*nonlinear=*/rewriter.getStringAttr("NONE"),
        /*nonlinear_in_scales=*/FloatAttr(),
        /*nonlinear_in_zeropoints=*/IntegerAttr(),
        /*prelu_in=*/IntegerAttr(),
        /*prelu_shift=*/IntegerAttr(),
        /*type=*/rewriter.getStringAttr("ADD"));

    rewriter.replaceOp(expandOp, fusedOp.getResult());
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
    // Pattern 5: Replace Expand with Eltwise ADD
    //========================================================================

    patterns.add<ReplaceExpandWithEltwise>(context);

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