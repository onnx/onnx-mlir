// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

// True when `op` keeps the same type on input (X) and output, i.e. it does not
// change scale/zero-point (no requantization). For quantized tensors a
// different scale/zero-point shows up as a different !quant.uniform type, so a
// type mismatch means the op is carrying a requant and must not be removed or
// folded away (that would silently drop the requant). Works for any unary op
// with X / result accessors (Relu, LeakyRelu).
template <typename OpTy>
static bool isNonRequantizing(OpTy op) {
  return op.getX().getType() == op.getResult().getType();
}

// LeakyReLU alpha (negative-side slope). ONNX default is 0.01 when absent.
static double getLeakyAlpha(ONNXLeakyReluOp op) {
  FloatAttr alphaAttr = op.getAlphaAttr();
  return alphaAttr ? alphaAttr.getValue().convertToDouble() : 0.01;
}

// A merged negative slope at or below this magnitude is treated as 0 (i.e.
// ReLU). Guards against numeric underflow of product(alpha) (e.g. 0.01^k) that
// the downstream fixed-point kernel parameters cannot represent. Heuristic --
// see constraint (6); changing it requires accuracy validation.
static constexpr double kLeakySlopeFlushToReluThreshold = 1e-6;

// True when `t` is a quantized type whose negative range is NOT representable:
// an unsigned quantization with the zero-point pinned at the minimum storage
// code (e.g. uint8 zp=0) maps every value to >= 0, so a leaky negative branch
// clamps to 0 there (effective slope 0). Constraint (3) negative-range guard.
static bool isUnsignedZeroPointAtMin(Type t) {
  auto shaped = dyn_cast<ShapedType>(t);
  if (!shaped)
    return false;
  auto qType = dyn_cast<quant::UniformQuantizedType>(shaped.getElementType());
  if (!qType || qType.isSigned())
    return false;
  return qType.getZeroPoint() == qType.getStorageTypeMin();
}

// Read a scalar (splat) constant as a real value. Handles plain float
// constants and ui8/i8 constants wrapped in !quant.uniform<...> (dequantized
// as (raw - zeroPoint) * scale). Returns std::nullopt otherwise.
static std::optional<double> readScalarConstant(Value v) {
  auto constOp = v.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return std::nullopt;
  auto valueAttr = constOp.getValue();
  if (!valueAttr)
    return std::nullopt;
  auto dense = dyn_cast<DenseElementsAttr>(*valueAttr);
  if (!dense || !dense.isSplat())
    return std::nullopt;

  Type elemType = dense.getType().getElementType();
  if (isa<FloatType>(elemType))
    return dense.getSplatValue<APFloat>().convertToDouble();

  auto resultType = dyn_cast<RankedTensorType>(constOp.getResult().getType());
  if (!resultType)
    return std::nullopt;
  auto qType =
      dyn_cast<quant::UniformQuantizedType>(resultType.getElementType());
  if (!qType || !isa<IntegerType>(elemType))
    return std::nullopt;
  int64_t raw = qType.isSigned() ? dense.getSplatValue<APInt>().getSExtValue()
                                 : dense.getSplatValue<APInt>().getZExtValue();
  return (static_cast<double>(raw) - qType.getZeroPoint()) * qType.getScale();
}

// True if `v` is provably >= 0 for every element because it comes from a
// non-negative-producing op:
//   * onnx.Relu                       -> output >= 0 by definition
//   * onnx.Clip with min >= 0         -> output >= min >= 0
//                                        (covers ReLU6 = Clip(x, 0, 6))
// Note: LeakyReLU is intentionally NOT here -- its negative branch (alpha*x)
// can be negative, so it does not establish non-negativity.
static bool isProvablyNonNegative(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<ONNXReluOp>(def))
    return true;
  if (auto clip = dyn_cast<ONNXClipOp>(def)) {
    Value minV = clip.getMin();
    if (!minV || isa<NoneType>(minV.getType()))
      return false; // no explicit lower bound
    std::optional<double> minVal = readScalarConstant(minV);
    return minVal.has_value() && *minVal >= 0.0;
  }
  return false;
}

/// Remove a redundant ONNXReluOp whose input is already non-negative.
///
/// Two sources of non-negativity are handled in one pattern:
///
///   (1) Relu(Relu(x))  ->  Relu(x)
///       Relu is idempotent. This keeps the original behavior: skip when the
///       inner Relu fans out, and rebuild this Relu on the inner input so the
///       ResultNamesUpdater listener transfers ResultNames onto the new op.
///       The greedy driver re-runs until the whole chain collapses.
///
///   (2) Relu(Clip(x, min, max)) with min >= 0  ->  Clip(x, min, max)
///       Clip output is >= min >= 0 (covers ReLU6 = Clip(x, 0, 6) and general
///       non-negative clamps), so Relu is a no-op and is replaced by its input.
///
/// For-now simplification: a Relu is only removed when it does not also
/// requantize. For quantized tensors a different scale/zero-point shows up as
/// a different !quant.uniform type, so a Relu whose input type differs from
/// its output type is carrying a requant; removing it would silently drop
/// that requant. We skip those until requant-preserving removal is handled.
struct RemoveRedundantReluPattern : public OpRewritePattern<ONNXReluOp> {
  using OpRewritePattern<ONNXReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReluOp reluOp, PatternRewriter &rewriter) const override {
    // Skip if this Relu also requantizes; removing it would drop the requant.
    if (!isNonRequantizing(reluOp))
      return failure();

    // (1) Relu(Relu(x)): keep the original one-use / rebuild-and-replace flow.
    if (auto prevRelu = reluOp.getX().getDefiningOp<ONNXReluOp>()) {
      // Skip fan-out: only collapse when the inner Relu feeds this Relu alone.
      if (!prevRelu->hasOneUse())
        return failure();
      // Skip if the inner Relu also requantizes; collapsing drops the requant.
      if (!isNonRequantizing(prevRelu))
        return failure();
      // Rebuild this Relu on the inner input and replace through the rewriter
      // so the ResultNamesUpdater listener transfers ResultNames onto new op.
      auto newRelu = rewriter.create<ONNXReluOp>(
          reluOp.getLoc(), reluOp.getType(), prevRelu.getX());
      rewriter.replaceOp(reluOp, newRelu.getResult());
      return success();
    }

    // (2) Relu after a Clip with min >= 0 (incl. ReLU6): input is already
    // non-negative, so this Relu is a no-op; replace it by its input.
    if (isProvablyNonNegative(reluOp.getX())) {
      rewriter.replaceOp(reluOp, reluOp.getX());
      return success();
    }

    return failure();
  }
};

/// Fold two adjacent LeakyReLUs:  leaky(a2, leaky(a1, x))  ->  leaky(a1*a2, x).
///
/// LeakyReLU is closed under composition: for x >= 0 both are the identity, and
/// for x < 0 the slopes multiply (L_a2 o L_a1 = L_{a1*a2}). The fold honors the
/// LeakyReLU cascade constraints (AIESW-35444 / report section 7.1):
///
///   (1) Activation identity: both ops are LeakyReLU with non-negative slope.
///       A negative slope would flip signs and break the composition proof.
///   (2) Slope merge: merged slope = a1 * a2.
///   (3) Negative-range guard: if the intermediate quantization cannot
///       represent the negative range (unsigned zero-point at the min code,
///       e.g. uint8 zp=0), the negative branch clamps to 0, so the effective
///       slope is 0 -> canonicalize to ReLU, not LeakyReLU(product).
///   (4) Quantization contract: only fold when the intermediate, input and
///       output quantization are identical (both ops non-requantizing), so the
///       removed intermediate is an identity requant; otherwise keep the ops.
///   (5) Fanout/output: only fold when the inner op feeds this op alone
///       (no fanout, not a graph output), so removing it is safe.
///   (6) Numeric underflow: if the merged slope underflows the kernel's
///       representable range, the negative branch becomes 0 -> ReLU.
///
/// PReLU (per-channel slopes) follows the same rule (elementwise slope product)
/// but is not handled here.
struct AdjacentLeakyReluFoldPattern : public OpRewritePattern<ONNXLeakyReluOp> {
  using OpRewritePattern<ONNXLeakyReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXLeakyReluOp op, PatternRewriter &rewriter) const override {
    auto inner = op.getX().getDefiningOp<ONNXLeakyReluOp>();
    if (!inner)
      return failure();

    // (5) Fanout/output guard: inner must feed this op alone.
    if (!inner->hasOneUse())
      return failure();

    // (4) Quantization contract: identical input/intermediate/output quant
    // (both ops non-requantizing) so the dropped intermediate is an identity.
    if (!isNonRequantizing(inner) || !isNonRequantizing(op))
      return failure();

    // (1) Activation identity: non-negative slopes only.
    double a1 = getLeakyAlpha(inner);
    double a2 = getLeakyAlpha(op);
    if (a1 < 0.0 || a2 < 0.0)
      return failure();

    // (2) Slope merge.
    double merged = a1 * a2;

    // (3) negative-range guard + (6) underflow: when the negative branch is
    // (or becomes) zero, fold to ReLU instead of LeakyReLU(product).
    bool negativeBranchIsZero =
        isUnsignedZeroPointAtMin(inner.getResult().getType()) ||
        merged <= kLeakySlopeFlushToReluThreshold;

    if (negativeBranchIsZero) {
      rewriter.replaceOpWithNewOp<ONNXReluOp>(
          op, op.getResult().getType(), inner.getX());
    } else {
      rewriter.replaceOpWithNewOp<ONNXLeakyReluOp>(op, op.getResult().getType(),
          inner.getX(), rewriter.getF32FloatAttr(static_cast<float>(merged)));
    }
    return success();
  }
};

/// Remove a redundant LeakyReLU whose input is provably non-negative
/// (produced by Relu or Clip(min>=0)): leaky(x) = x for x >= 0, so it is the
/// identity and is replaced by its input.
///
/// Same requant guard: skipped when the LeakyReLU also requantizes (input
/// scale/zero-point differs from output), since removing it would drop the
/// requant.
struct RedundantLeakyReluEliminationPattern
    : public OpRewritePattern<ONNXLeakyReluOp> {
  using OpRewritePattern<ONNXLeakyReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXLeakyReluOp op, PatternRewriter &rewriter) const override {
    if (!isNonRequantizing(op))
      return failure();
    if (!isProvablyNonNegative(op.getX()))
      return failure();
    rewriter.replaceOp(op, op.getX());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct RemoveRedundantReluLikeOpsPass
    : public PassWrapper<RemoveRedundantReluLikeOpsPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "remove-redundant-relu-like-ops";
  }
  StringRef getDescription() const override {
    return "Eliminate redundant Relu/LeakyRelu ops: collapse Relu chains, "
           "drop Relu/LeakyRelu on provably non-negative inputs, and fold "
           "adjacent LeakyRelu cascades into a single LeakyRelu.";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RemoveRedundantReluPattern>(context);
    patterns.add<AdjacentLeakyReluFoldPattern>(context);
    patterns.add<RedundantLeakyReluEliminationPattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createRemoveRedundantReluLikeOpsPass() {
  return std::make_unique<RemoveRedundantReluLikeOpsPass>();
}

} // namespace onnx_mlir
