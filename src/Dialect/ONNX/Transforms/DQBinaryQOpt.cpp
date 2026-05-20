//===- foldDqBinaryQPattern.cpp - Remove DQ-Binary-Q chains -----*- C++ -*-===//
//
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/STLExtras.h"
#include <cmath>
#include <limits>
#include <optional>
#include <variant>

using namespace mlir;
using namespace onnx_mlir;

namespace {

template <typename T>
std::optional<T> getScalarTensorValue(ONNXConstantOp constOp) {
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(constOp.getValueAttr());
  if (!elementsAttr)
    return std::nullopt;

  Type elementType = elementsAttr.getElementType();

  // Fast path: splat
  if (elementsAttr.isSplat()) {
    if (mlir::isa<FloatType>(elementType)) {
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
        APFloat splatValue = elementsAttr.getSplatValue<APFloat>();
        return static_cast<T>(splatValue.convertToDouble());
      }
    }
    if (auto intType = mlir::dyn_cast<IntegerType>(elementType)) {
      if constexpr (std::is_integral_v<T>) {
        APInt splatValue = elementsAttr.getSplatValue<APInt>();
        if (intType.isUnsigned())
          return static_cast<T>(splatValue.getZExtValue());
        else
          return static_cast<T>(splatValue.getSExtValue());
      }
    }
    return std::nullopt;
  }

  // Non‑splat case: check rank
  auto shapedTy = mlir::dyn_cast<ShapedType>(elementsAttr.getType());
  if (!shapedTy || !shapedTy.hasStaticShape())
    return std::nullopt;

  // Case: rank 0 → scalar element directly
  if (shapedTy.getRank() == 0) {
    auto firstAttr = *elementsAttr.getValues<Attribute>().begin();
    if (auto fAttr = mlir::dyn_cast<FloatAttr>(firstAttr)) {
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
        return static_cast<T>(fAttr.getValueAsDouble());
    }
    if (auto iAttr = mlir::dyn_cast<IntegerAttr>(firstAttr)) {
      if constexpr (std::is_integral_v<T>)
        return static_cast<T>(iAttr.getInt()); // signed ok
    }
    return std::nullopt;
  }

  // Case: rank >= 1 → flatten & check all the same
  std::set<double> flattenedFP;
  std::set<int64_t> flattenedInt;

  if (mlir::isa<FloatType>(elementType)) {
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
      for (auto a : elementsAttr.getValues<FloatAttr>())
        flattenedFP.insert(a.getValueAsDouble());
      if (flattenedFP.size() == 1)
        return static_cast<T>(*flattenedFP.begin());
    }
  } else if (auto intType = mlir::dyn_cast<IntegerType>(elementType)) {
    if constexpr (std::is_integral_v<T>) {
      for (auto a : elementsAttr.getValues<IntegerAttr>())
        flattenedInt.insert(intType.isUnsigned() ? a.getUInt() : a.getInt());
      if (flattenedInt.size() == 1)
        return static_cast<T>(*flattenedInt.begin());
    }
  }
  return std::nullopt;
}

template <typename T>
std::optional<T> getScalarTensorValueFromVal(Value value) {
  if (!value) {
    return std::nullopt;
  }
  auto constOp = value.getDefiningOp<ONNXConstantOp>();
  if (!constOp) {
    return std::nullopt;
  }
  return getScalarTensorValue<T>(constOp);
}

static mlir::DenseElementsAttr makeScalarDEA(
    mlir::ShapedType likeTy, double d) {
  using namespace mlir;

  auto ranked = mlir::dyn_cast<RankedTensorType>(likeTy);
  if (!ranked || !ranked.hasStaticShape() || ranked.getNumElements() != 1)
    return {};

  Type outET = ranked.getElementType();

  // If target is float, just create a float attr with outET semantics.
  if (auto outFT = mlir::dyn_cast<FloatType>(outET)) {
    // Round in the semantics of outET if it's float; otherwise just use d.
    double dv = d;
    if (auto useFT = mlir::dyn_cast<FloatType>(outET)) {
      // Convert through APFloat with 'outET' semantics, then to double.
      llvm::APFloat ap(d);
      bool loses = false;
      ap.convert(useFT.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
          &loses);
      dv = ap.convertToDouble();
    }
    return DenseElementsAttr::get(ranked, FloatAttr::get(outFT, dv));
  }

  // If target is integer, round+clamp as per 'outET' (if integer), then emit as
  // outET.
  if (auto outIT = mlir::dyn_cast<IntegerType>(outET)) {
    // Decide signedness/width for clamping from outET if it's integer, else
    // from outET.
    IntegerType clampIT =
        mlir::isa<IntegerType>(outET) ? mlir::cast<IntegerType>(outET) : outIT;

    int64_t iv = static_cast<int64_t>(std::llround(d));
    const unsigned bw = clampIT.getWidth();
    const bool isSigned = clampIT.isSigned() || clampIT.isSignless();

    const int64_t minV = isSigned ? (-(int64_t(1) << (bw - 1))) : 0;
    const int64_t maxV =
        isSigned ? ((int64_t(1) << (bw - 1)) - 1) : ((int64_t(1) << bw) - 1);
    iv = std::min<int64_t>(std::max<int64_t>(iv, minV), maxV);

    // This guarantees the result type matches `likeTy`.
    if (outIT.isSigned()) {
      // For signed out type, encode iv as signed.
      return DenseElementsAttr::get(ranked, IntegerAttr::get(outIT, iv));
    } else {
      // For unsigned out type, encode iv as unsigned (mask to width).
      uint64_t u = static_cast<uint64_t>(iv);
      if (outIT.getWidth() < 64)
        u &= ((uint64_t(1) << outIT.getWidth()) - 1);
      return DenseElementsAttr::get(ranked, IntegerAttr::get(outIT, u));
    }
  }

  return {};
}

static void updateInitializer(mlir::PatternRewriter &rewriter,
    mlir::Operation *targetOp, mlir::Value oldInit, double newScalar) {
  using namespace mlir;

  if (!targetOp || !oldInit)
    return;

  auto oldCst = oldInit.getDefiningOp<ONNXConstantOp>();
  if (!oldCst)
    return;

  auto likeTy = mlir::dyn_cast<ShapedType>(oldInit.getType());
  if (!likeTy || !likeTy.hasStaticShape() || likeTy.getNumElements() != 1)
    return;

  DenseElementsAttr payload = makeScalarDEA(likeTy, newScalar);
  if (!payload)
    return;

  // Check for single-use by targetOp.
  auto singleUseByTarget = [&]() -> bool {
    auto it = oldInit.use_begin(), e = oldInit.use_end();
    if (it == e)
      return false;
    auto *owner = it->getOwner();
    ++it;
    return (it == e) && (owner == targetOp);
  };

  if (singleUseByTarget()) {
    rewriter.modifyOpInPlace(oldCst, [&] {
      oldCst->setAttr("value", payload);
      oldCst->removeAttr("sparse_value");
      oldCst->removeAttr("value_float");
      oldCst->removeAttr("value_floats");
      oldCst->removeAttr("value_int");
      oldCst->removeAttr("value_ints");
      oldCst->removeAttr("value_string");
      oldCst->removeAttr("value_strings");
    });
    return;
  }

  // Multi-use: clone a fresh constant with same type as oldInit.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(targetOp);

  OperationState st(targetOp->getLoc(), ONNXConstantOp::getOperationName());
  st.addTypes(likeTy);
  st.addAttribute("value", payload);

  Operation *raw = Operation::create(st);
  rewriter.insert(raw);

  auto newCst = llvm::dyn_cast<ONNXConstantOp>(raw);
  if (!newCst)
    return;

  // Replace only the operand equal to oldInit.
  for (unsigned i = 0, e = targetOp->getNumOperands(); i < e; ++i) {
    if (targetOp->getOperand(i) == oldInit) {
      targetOp->setOperand(i, newCst.getOutput());
      break;
    }
  }
}

// Returns success() iff Q->DQ is *removable* under strict checks.
// If doRewrite==true, it also *applies* the rewrite for this DQ (replaces DQ
// with Q.x).
static mlir::LogicalResult tryRemoveQThenDQChain(
    mlir::PatternRewriter &rewriter, mlir::ONNXDequantizeLinearOp dqOp,
    bool doRewrite) {
  using namespace mlir;

  // Match direct Q -> DQ
  auto qOp = dqOp.getX().template getDefiningOp<ONNXQuantizeLinearOp>();
  if (!qOp)
    return failure();

  // 1) Axis / block_size must match
  if (qOp.getAxis() != dqOp.getAxis())
    return failure();
  if (qOp.getBlockSize() != dqOp.getBlockSize())
    return failure();

  // 2) Zero-points must match scalars/splats
  auto zpQ = getElementAttributeFromONNXValue(qOp.getYZeroPoint());
  auto zpDQ = getElementAttributeFromONNXValue(dqOp.getXZeroPoint());
  if (!zpQ || !zpDQ || zpQ != zpDQ)
    return failure();

  // 3) Scales must match scalars/splats
  auto sQ = getElementAttributeFromONNXValue(qOp.getYScale());
  auto sDQ = getElementAttributeFromONNXValue(dqOp.getXScale());
  if (!sQ || !sDQ || sQ != sDQ)
    return failure();

  // 4) Element type parity between Q.x and DQ.y
  auto qInTypeOp = qOp.getX().getType();
  auto dqOutTypeOp = dqOp.getResult().getType();
  auto qInT = mlir::dyn_cast<TensorType>(qInTypeOp);
  auto dqOutT = mlir::dyn_cast<TensorType>(dqOutTypeOp);
  if (!qInT || !dqOutT)
    return failure();
  if (dqOutT.getElementType() != qInT.getElementType())
    return failure();

  // If only checking removability, stop here.
  if (!doRewrite)
    return success();

  // Rewrite: replace DQ with Q's float input.
  rewriter.replaceOp(dqOp, qOp.getX());
  return success();
}

// If doRewrite=false: returns true iff *any* removable DQ user exists (no
// mutation). If doRewrite=true: performs removals and returns true iff it
// removed at least one DQ. Also erases Q if it becomes dead after removals.
static bool Remove_Q_Plus_DQ(
    mlir::PatternRewriter &rewriter, ONNXQuantizeLinearOp qOp, bool doRewrite) {
  using namespace mlir;
  if (!qOp)
    return false;

  if (doRewrite) {
    // MATCH PHASE: Collect all removable DQ nodes
    SmallVector<ONNXDequantizeLinearOp, 4> matching_dequantize_nodes;

    for (Operation *user : qOp.getY().getUsers()) {
      if (auto tailDQ = llvm::dyn_cast<ONNXDequantizeLinearOp>(user)) {
        if (succeeded(
                tryRemoveQThenDQChain(rewriter, tailDQ, /*doRewrite*/ false))) {
          matching_dequantize_nodes.push_back(tailDQ);
        }
      }
    }

    // Early exit if nothing to remove
    if (matching_dequantize_nodes.empty())
      return false;

    // MODIFY PHASE: Perform actual removals
    for (auto dqOp : matching_dequantize_nodes) {
      (void)tryRemoveQThenDQChain(rewriter, dqOp, /*doRewrite*/ true);
    }

    // CLEANUP PHASE: Remove Q if it has no remaining users
    if (qOp->use_empty()) {
      rewriter.eraseOp(qOp);
    }

    return true;

  } else {
    // CHECK MODE: Only check if removable without modifying
    int removableCount = 0;

    for (Operation *user : qOp.getY().getUsers()) {
      if (auto tailDQ = llvm::dyn_cast<ONNXDequantizeLinearOp>(user)) {
        if (succeeded(
                tryRemoveQThenDQChain(rewriter, tailDQ, /*doRewrite*/ false))) {
          ++removableCount;
        }
      }
    }

    return removableCount > 0;
  }
}

static bool isValuePreservingOp(mlir::Operation *op) {
  if (!op)
    return false;
  return isa<mlir::ONNXIdentityOp, mlir::ONNXReshapeOp, mlir::ONNXSqueezeOp,
      mlir::ONNXUnsqueezeOp, mlir::ONNXTransposeOp>(op);
}

template <typename BinOp>
struct FoldBinaryThroughQDQ : public OpRewritePattern<BinOp> {
  FoldBinaryThroughQDQ(MLIRContext *context)
      : OpRewritePattern<BinOp>(context) {}

private:
  struct MatchState {
    ONNXDequantizeLinearOp dequantActivationOfBinOp =
        nullptr;                                       // BinaryOP parent op
    ONNXQuantizeLinearOp quantOutputOfBinOp = nullptr; // BinaryOp child op

    // Destination/source ops picked by find_destination_node()
    mlir::Operation *dstNode =
        nullptr; // DQ when folding into DQ, or Q when folding into Q
    mlir::Operation *srcNode = nullptr;

    // Current destination params (read before fold)
    mlir::Value dstScaleValue;     // Scale Value for reuse
    mlir::Value dstZeroPointValue; // Zero point Value for reuse
    double dstScale = 0.0;
    int64_t dstZeroPoint = 0;

    // New params to write after fold
    double newScale = 0.0;
    int64_t newZp = 0;

    // Constant value folded
    double kValue = 0.0;
  };

  LogicalResult match_qdq(mlir::PatternRewriter &rewriter, MatchState &state,
      ONNXDequantizeLinearOp dq1, ONNXDequantizeLinearOp dq2,
      BinOp binaryOp) const {

    ONNXDequantizeLinearOp constantDqOp = nullptr;
    ONNXConstantOp constantSourceOp = nullptr;
    bool constantIsFirstOperand = false;

    // Case 1: Direct ConstantOp as input to the DQ.
    if (auto constOp = dq1.getX().getDefiningOp<ONNXConstantOp>()) {
      constantDqOp = dq1;
      state.dequantActivationOfBinOp = dq2;
      constantSourceOp = constOp;
      constantIsFirstOperand = true;
    } else if (auto constOp = dq2.getX().getDefiningOp<ONNXConstantOp>()) {
      constantDqOp = dq2;
      state.dequantActivationOfBinOp = dq1;
      constantSourceOp = constOp;
    }
    // Case 2: The input to the DQ op comes from a chain whose input is a
    // constant.
    else {
      if (auto intermediateOp = dq1.getX().getDefiningOp()) {
        if (isValuePreservingOp(intermediateOp)) {
          if (auto constOp = intermediateOp->getOperand(0)
                  .getDefiningOp<ONNXConstantOp>()) {
            constantDqOp = dq1;
            state.dequantActivationOfBinOp = dq2;
            constantSourceOp = constOp;
            constantIsFirstOperand = true;
          }
        }
      }
      if (auto intermediateOp = dq2.getX().getDefiningOp()) {
        if (isValuePreservingOp(intermediateOp)) {
          if (auto constOp = intermediateOp->getOperand(0)
                  .getDefiningOp<ONNXConstantOp>()) {
            constantDqOp = dq2;
            state.dequantActivationOfBinOp = dq1;
            constantSourceOp = constOp;
          }
        }
      }
    }

    // Case 3: The input to the DQ op comes from a Q->DQ chain where Q's input
    // is a constant.
    bool isConstantFromQDQChain = false;
    if (!constantDqOp || !constantSourceOp) {
      // Check dq1: Constant -> Q -> DQ
      if (auto qOp = dq1.getX().getDefiningOp<ONNXQuantizeLinearOp>()) {
        if (succeeded(
                tryRemoveQThenDQChain(rewriter, dq1, /*doRewrite=*/false))) {
          if (auto constOp = qOp.getX().getDefiningOp<ONNXConstantOp>()) {
            constantDqOp = dq1;
            state.dequantActivationOfBinOp = dq2;
            constantSourceOp = constOp;
            constantIsFirstOperand = true;
            isConstantFromQDQChain = true;
          }
        }
      }
      // Check dq2: Constant -> Q -> DQ
      if (!constantDqOp || !constantSourceOp) {
        if (auto qOp = dq2.getX().getDefiningOp<ONNXQuantizeLinearOp>()) {
          if (succeeded(
                  tryRemoveQThenDQChain(rewriter, dq2, /*doRewrite=*/false))) {
            if (auto constOp = qOp.getX().getDefiningOp<ONNXConstantOp>()) {
              constantDqOp = dq2;
              state.dequantActivationOfBinOp = dq1;
              constantSourceOp = constOp;
              isConstantFromQDQChain = true;
            }
          }
        }
      }
    }

    if (!constantDqOp || !constantSourceOp || !state.dequantActivationOfBinOp) {
      return rewriter.notifyMatchFailure(binaryOp,
          "Remove binary op only if one of the dequantize linear "
          "input has const scalar value");
    }

    // Check: Div and Sub are not supported when weight is the first input
    if (constantIsFirstOperand &&
        (llvm::isa<ONNXSubOp>(binaryOp) || llvm::isa<ONNXDivOp>(binaryOp))) {
      return rewriter.notifyMatchFailure(binaryOp,
          "Qdq initializer: Div and Sub are not supported when "
          "weight is the first input");
    }

    // Find kvalue and store scale_dtype and zeroPointDtype
    {
      // When constant comes from a Q->DQ chain, it's in float form
      // When constant comes directly to DQ, it's in quantized (int) form
      if (isConstantFromQDQChain) {
        // Constant is in float form: just use the value directly
        auto scalar_value_opt = getScalarTensorValue<double>(constantSourceOp);
        if (!scalar_value_opt) {
          return rewriter.notifyMatchFailure(constantSourceOp,
              " must be a scalar value or a list of same value");
        }
        state.kValue = *scalar_value_opt;
      } else {
        // Constant is in quantized form: need to dequantize it
        auto scalar_value_opt = getScalarTensorValue<int64_t>(constantSourceOp);
        if (!scalar_value_opt) {
          return rewriter.notifyMatchFailure(constantSourceOp,
              " must be a scalar value or a list of same value");
        }
        Value scaleVal = constantDqOp.getXScale();
        Value zpVal = constantDqOp.getXZeroPoint();
        auto scale_value_opt = getScalarTensorValueFromVal<double>(scaleVal);
        auto zp_value_opt = getScalarTensorValueFromVal<int64_t>(zpVal);
        if (!scale_value_opt || !zp_value_opt) {
          return rewriter.notifyMatchFailure(
              constantDqOp, " must be a scalar value or a list of same value");
        }
        // Calculate and store kValue.
        state.kValue = (*scalar_value_opt - *zp_value_opt) * *scale_value_opt;
      }
    }
    return success();
  }

  LogicalResult match_binary_op(mlir::PatternRewriter &rewriter,
      MatchState &state, BinOp binaryOp) const {
    ONNXConstantOp constantOp = nullptr;

    Value lhs = binaryOp.getOperand(0);
    Value rhs = binaryOp.getOperand(1);
    Value out = binaryOp->getResult(0);
    state.quantOutputOfBinOp =
        mlir::dyn_cast<ONNXQuantizeLinearOp>(*out.getUsers().begin());

    // -------- Case A: lhs is DQ, rhs is Constant --------
    if (auto dqOp = lhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
      if (auto constOp = rhs.getDefiningOp<ONNXConstantOp>()) {
        state.dequantActivationOfBinOp = dqOp;
        constantOp = constOp;
      }
    }
    // -------- Case A reversed --------
    else if (auto dqOp = rhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
      if (auto constOp = lhs.getDefiningOp<ONNXConstantOp>()) {
        // Check: Div and Sub are not supported when weight is the first input
        if (llvm::isa<ONNXSubOp>(binaryOp) || llvm::isa<ONNXDivOp>(binaryOp)) {
          return rewriter.notifyMatchFailure(binaryOp,
              "non-qdq initializer: Div and Sub are not supported "
              "when weight is the first input");
        }
        state.dequantActivationOfBinOp = dqOp;
        constantOp = constOp;
      }
    }

    // -------- Fill state values for Case A and Case A reversed --------
    if (state.dequantActivationOfBinOp && constantOp) {
      auto kValueOpt = getScalarTensorValue<double>(constantOp);
      if (!kValueOpt) {
        return rewriter.notifyMatchFailure(
            constantOp, " must be a scalar value or a list of same value");
      }
      state.kValue = kValueOpt.value();
      return success();
    }

    // -------- Case B: both inputs are DQ --------
    auto dqOp1 = lhs.getDefiningOp<ONNXDequantizeLinearOp>();
    auto dqOp2 = rhs.getDefiningOp<ONNXDequantizeLinearOp>();

    if (dqOp1 && dqOp2) {
      return match_qdq(rewriter, state, dqOp1, dqOp2, binaryOp);
    }
    return failure();
  }

  LogicalResult check_needed_values(mlir::PatternRewriter &rewriter,
      const MatchState &state, Operation *binaryOp) const {
    const bool dstIsDQ = llvm::isa<ONNXDequantizeLinearOp>(state.dstNode);
    const bool dstIsQ = llvm::isa<ONNXQuantizeLinearOp>(state.dstNode);

    // scale_new = scale / k  (for Div when folding into DQ)  OR  scale_new =
    // scale / k (for Mul when folding into Q)
    // Avoid division by zero when k == 0 in those cases.
    if (state.kValue == 0.0) {
      if (dstIsDQ && llvm::isa<ONNXDivOp>(binaryOp)) {
        return rewriter.notifyMatchFailure(binaryOp,
            "when opType is Div, remove binary op only if k_value is not zero, "
            "to avoid ZeroDivisionError");
      } else if (dstIsQ && llvm::isa<ONNXMulOp>(binaryOp)) {
        return rewriter.notifyMatchFailure(binaryOp,
            "when opType is Mul, remove binary op only if k_value is not zero, "
            "to avoid ZeroDivisionError");
      }
    }

    // k/scale is used for Add/Sub to update zero_point.
    // Avoid division by zero when dstScale == 0.
    if (state.dstScale == 0.0 && (llvm::isa<ONNXAddOp, ONNXSubOp>(binaryOp))) {
      return rewriter.notifyMatchFailure(binaryOp,
          "when opType is Add or Sub, remove binary op only if scale is not "
          "zero, to avoid ZeroDivisionError");
    }

    return mlir::success();
  }

  static bool compute_new_scale_and_zp_values(MatchState &state) {
    double newScale = state.dstScale;
    double newZpFloat = static_cast<double>(state.dstZeroPoint);
    const double kVal = state.kValue;
    const bool dstIsDQ = llvm::isa<ONNXDequantizeLinearOp>(state.dstNode);

    if constexpr (std::is_same_v<BinOp, ONNXAddOp>) {
      if (dstIsDQ)
        newZpFloat -= (kVal / newScale);
      else
        newZpFloat += (kVal / newScale);
    } else if constexpr (std::is_same_v<BinOp, ONNXSubOp>) {
      if (dstIsDQ)
        newZpFloat += (kVal / newScale);
      else
        newZpFloat -= (kVal / newScale);
    } else if constexpr (std::is_same_v<BinOp, ONNXMulOp>) {
      if (dstIsDQ)
        newScale *= kVal;
      else
        newScale /= kVal;
    } else if constexpr (std::is_same_v<BinOp, ONNXDivOp>) {
      if (dstIsDQ)
        newScale /= kVal;
      else
        newScale *= kVal;
    } else {
      static_assert(std::is_same_v<BinOp, ONNXAddOp> ||
                        std::is_same_v<BinOp, ONNXSubOp> ||
                        std::is_same_v<BinOp, ONNXMulOp> ||
                        std::is_same_v<BinOp, ONNXDivOp>,
          "Unsupported binary operation type for this pattern");
      return false;
    }

    int64_t newZp = (newZpFloat >= 0.0) ? (int64_t)std::floor(newZpFloat)
                                        : (int64_t)std::ceil(newZpFloat);
    state.newScale = newScale;
    state.newZp = newZp;

    return true;
  }

  LogicalResult checkNewQDQParameterFits(
      mlir::PatternRewriter &rewriter, const MatchState &state) const {
    using namespace mlir;

    if (!state.dstNode)
      return rewriter.notifyMatchFailure(
          state.dstNode, "dstNode is null in checkNewQDQParameterFits");

    // Reuse scale and zero point values already stored in state
    Value scaleVal = state.dstScaleValue;
    Value zpVal = state.dstZeroPointValue;

    // Extract element types
    auto scaleType = mlir::dyn_cast<ShapedType>(scaleVal.getType());
    auto zpType = mlir::dyn_cast<ShapedType>(zpVal.getType());

    if (!scaleType || !zpType)
      return rewriter.notifyMatchFailure(
          state.dstNode, "scale or zero point is not a shaped type");

    Type scaleElemType = scaleType.getElementType();
    Type zpElemType = zpType.getElementType();

    // Check zero point range
    if (auto zpIntType = mlir::dyn_cast<IntegerType>(zpElemType)) {
      int64_t zpMin, zpMax;
      unsigned bitWidth = zpIntType.getWidth();

      // Handle special cases for int4/uint4
      if (bitWidth == 4) {
        if (zpIntType.isUnsigned()) {
          zpMin = 0;
          zpMax = 15;
        } else {
          zpMin = -8;
          zpMax = 7;
        }
      } else {
        // Standard integer types
        if (zpIntType.isUnsigned()) {
          zpMin = 0;
          zpMax = (bitWidth == 64) ? INT64_MAX : ((int64_t(1) << bitWidth) - 1);
        } else {
          zpMin =
              (bitWidth == 64) ? INT64_MIN : (-(int64_t(1) << (bitWidth - 1)));
          zpMax = (bitWidth == 64) ? INT64_MAX
                                   : ((int64_t(1) << (bitWidth - 1)) - 1);
        }
      }

      if (state.newZp < zpMin || state.newZp > zpMax) {
        return rewriter.notifyMatchFailure(state.dstNode,
            ("New zero point value " + std::to_string(state.newZp) +
                " cannot fit in " + std::to_string(bitWidth) + "-bit " +
                (zpIntType.isUnsigned() ? "unsigned" : "signed") +
                " integer type (range: [" + std::to_string(zpMin) + ", " +
                std::to_string(zpMax) + "])")
                .c_str());
      }
    }

    // Check scale range
    if (auto scaleFloatType = mlir::dyn_cast<FloatType>(scaleElemType)) {
      double scaleMin, scaleMax;

      // Determine range based on float type
      if (scaleFloatType.isF16()) {
        scaleMin = -65504.0;
        scaleMax = 65504.0;
      } else if (scaleFloatType.isBF16()) {
        scaleMin = -std::numeric_limits<float>::max(); // ≈ -3.39e38
        scaleMax = std::numeric_limits<float>::max();  // ≈ +3.39e38
      } else if (scaleFloatType.isF32()) {
        scaleMin = -std::numeric_limits<float>::max();
        scaleMax = std::numeric_limits<float>::max();
      } else if (scaleFloatType.isF64()) {
        scaleMin = -std::numeric_limits<double>::max();
        scaleMax = std::numeric_limits<double>::max();
      } else {
        // Unknown float type, cannot validate safely
        return rewriter.notifyMatchFailure(
            state.dstNode, "Unsupported float type for scale validation");
      }

      if (state.newScale < scaleMin || state.newScale > scaleMax) {
        std::string floatTypeName;
        if (scaleFloatType.isF16())
          floatTypeName = "float16";
        else if (scaleFloatType.isBF16())
          floatTypeName = "bfloat16";
        else if (scaleFloatType.isF32())
          floatTypeName = "float32";
        else if (scaleFloatType.isF64())
          floatTypeName = "float64";
        else
          floatTypeName = "unknown float";

        return rewriter.notifyMatchFailure(state.dstNode,
            ("New scale value " + std::to_string(state.newScale) +
                " cannot fit in " + floatTypeName + " type (range: [" +
                std::to_string(scaleMin) + ", " + std::to_string(scaleMax) +
                "])")
                .c_str());
      }
    }

    return success();
  }

  LogicalResult findDestinationNode(
      mlir::PatternRewriter &rewriter, MatchState &state, Operation *op) const {
    auto dq = state.dequantActivationOfBinOp;
    if (!dq)
      return rewriter.notifyMatchFailure(
          op, "dequantActivationOfBinOp not set in MatchState");

    auto q = dq.getX().template getDefiningOp<ONNXQuantizeLinearOp>();
    bool removableQDQ = false;
    if (q)
      removableQDQ = Remove_Q_Plus_DQ(rewriter, q, /*doRewrite=*/false);

    // Branch detection on distinct users
    auto hasBranchOnValue = [](mlir::Value v) {
      llvm::SmallPtrSet<mlir::Operation *, 8> uniq;
      for (auto *u : v.getUsers())
        uniq.insert(u);
      return uniq.size() > 1;
    };

    // Check if DQ output has multiple readers AND DQ input is Q
    const bool branch_after = q && hasBranchOnValue(dq.getY());

    // If there is branching just above the dequant (on DQ's input), the Q/DQ
    // removal at the end will also remove Q/DQ on other branches, so treat as
    // branch
    const bool branch_before = hasBranchOnValue(dq.getX());
    const bool branch_on_dequant_activation = branch_after || branch_before;

    // If we cannot remove Q->DQ (or there is branching), fold into DQ
    if (!removableQDQ || branch_on_dequant_activation) {
      state.dstNode = dq.getOperation();                       // DQ
      state.srcNode = state.quantOutputOfBinOp.getOperation(); // Q after binop

      state.dstScaleValue = dq.getXScale();
      state.dstZeroPointValue = dq.getXZeroPoint();
      auto scaleOpt = getScalarTensorValueFromVal<double>(state.dstScaleValue);
      auto zpOpt =
          getScalarTensorValueFromVal<int64_t>(state.dstZeroPointValue);
      if (!scaleOpt || !zpOpt)
        return rewriter.notifyMatchFailure(
            dq, "DQ x_scale/x_zero_point must be scalar");
      state.dstScale = *scaleOpt;
      state.dstZeroPoint = *zpOpt;
      return success();
    }

    // Else: fold into the Quantize after the binop
    auto qOut = state.quantOutputOfBinOp;
    if (!qOut)
      return rewriter.notifyMatchFailure(
          op, "expected a unique Quantize user of the binary result");

    state.dstNode = qOut.getOperation();                           // Q
    state.srcNode = state.dequantActivationOfBinOp.getOperation(); // DQ

    state.dstScaleValue = qOut.getYScale();
    state.dstZeroPointValue = qOut.getYZeroPoint();
    auto scaleOpt = getScalarTensorValueFromVal<double>(state.dstScaleValue);
    auto zpOpt = getScalarTensorValueFromVal<int64_t>(state.dstZeroPointValue);
    if (!scaleOpt || !zpOpt)
      return rewriter.notifyMatchFailure(
          qOut, "Quantize y_scale/y_zero_point must be scalar");
    state.dstScale = *scaleOpt;
    state.dstZeroPoint = *zpOpt;
    return success();
  }

public:
  LogicalResult matchAndRewrite(
      BinOp op, PatternRewriter &rewriter) const override {

    // STEP 1: Match begin: Assuming only one user
    if (!op->hasOneUse()) {
      return rewriter.notifyMatchFailure(op, "pattern requires a single user");
    }
    auto quantOutputOfBinOp =
        mlir::dyn_cast<ONNXQuantizeLinearOp>(*op->user_begin());
    if (!quantOutputOfBinOp) {
      return rewriter.notifyMatchFailure(
          op, "expected user to be an ONNXQuantizeLinearOp");
    }

    // Instantiate the state struct
    MatchState state;

    // STEP 2
    if (failed(match_binary_op(rewriter, state, op))) {
      return rewriter.notifyMatchFailure(op,
          " does not match to critieria to remove binary. Remove binary op "
          "only if one of the dequantize linear input "
          "has const scalar value ");
    }

    // STEP 3
    if (failed(findDestinationNode(rewriter, state, op))) {
      return failure();
    }

    // STEP 4
    if (failed(check_needed_values(rewriter, state, op))) {
      return failure();
    }

    // STEP 5: Compute new scale and zero point values
    if (!compute_new_scale_and_zp_values(state)) {
      return failure();
    }

    // STEP 5.5: Check that new values fit in destination dtypes (avoid
    // overflow/underflow)
    if (failed(checkNewQDQParameterFits(rewriter, state))) {
      return failure();
    }

    // STEP 6: Update initializer based on the binary op
    // Use the scale/zero-point Values already stored in state
    {
      if (!state.dstNode)
        return rewriter.notifyMatchFailure(op, "dstNode not set");

      if constexpr (std::is_same_v<BinOp, ONNXAddOp> ||
                    std::is_same_v<BinOp, ONNXSubOp>) {
        // Update zero-point (for Add/Sub)
        updateInitializer(rewriter, state.dstNode, state.dstZeroPointValue,
            static_cast<double>(state.newZp));
      } else if constexpr (std::is_same_v<BinOp, ONNXMulOp> ||
                           std::is_same_v<BinOp, ONNXDivOp>) {
        // Update scale (for Mul/Div)
        updateInitializer(
            rewriter, state.dstNode, state.dstScaleValue, state.newScale);
      }
    }

    // STEP 7: Remove binary op
    rewriter.replaceOp(op, state.dequantActivationOfBinOp.getResult());

    // STEP 8: Remove Q->DQ chain
    ONNXQuantizeLinearOp chainStartQ = nullptr;

    if (llvm::isa<ONNXDequantizeLinearOp>(state.dstNode)) {
      // Folding happened in the Dequantize: chain start is the Quantize after
      // the BinOp
      chainStartQ = state.quantOutputOfBinOp; // set earlier during match
    } else if (llvm::isa<ONNXQuantizeLinearOp>(state.dstNode)) {
      // Folding happened in the Quantize: chain start is the Quantize feeding
      // DQ.x
      if (auto dqAct = state.dequantActivationOfBinOp) {
        chainStartQ =
            dqAct.getX().template getDefiningOp<ONNXQuantizeLinearOp>();
      }
    }
    // Run the cleanup if we found a Quantize
    if (chainStartQ) {
      (void)Remove_Q_Plus_DQ(rewriter, chainStartQ, /*doRewrite=*/true);
    }

    return success();
  }
};

} // namespace

namespace onnx_mlir {

void getDQBinaryQPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldBinaryThroughQDQ<ONNXDivOp>>(context);
  patterns.add<FoldBinaryThroughQDQ<ONNXSubOp>>(context);
  patterns.add<FoldBinaryThroughQDQ<ONNXMulOp>>(context);
  patterns.add<FoldBinaryThroughQDQ<ONNXAddOp>>(context);
}

} // namespace onnx_mlir
