//===- DQBinaryQOptPass.cpp - XMC pass for DQ-Binary-Q folding ---*- C++
//-*-===//
//
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Standalone XMC pass that folds scalar binary operations (Add, Sub, Mul, Div)
// sandwiched between DequantizeLinear and QuantizeLinear into the quantization
// parameters (scale / zero-point), eliminating the binary op entirely.
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
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
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

  // Non-splat case: check rank
  auto shapedTy = mlir::dyn_cast<ShapedType>(elementsAttr.getType());
  if (!shapedTy || !shapedTy.hasStaticShape())
    return std::nullopt;

  // Case: rank 0 -> scalar element directly
  if (shapedTy.getRank() == 0) {
    auto firstAttr = *elementsAttr.getValues<Attribute>().begin();
    if (auto fAttr = mlir::dyn_cast<FloatAttr>(firstAttr)) {
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
        return static_cast<T>(fAttr.getValueAsDouble());
    }
    if (auto iAttr = mlir::dyn_cast<IntegerAttr>(firstAttr)) {
      if constexpr (std::is_integral_v<T>)
        return static_cast<T>(iAttr.getInt());
    }
    return std::nullopt;
  }

  // Case: rank >= 1 -> flatten & check all the same
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

  if (auto outFT = mlir::dyn_cast<FloatType>(outET)) {
    double dv = d;
    if (auto useFT = mlir::dyn_cast<FloatType>(outET)) {
      llvm::APFloat ap(d);
      bool loses = false;
      ap.convert(useFT.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
          &loses);
      dv = ap.convertToDouble();
    }
    return DenseElementsAttr::get(ranked, FloatAttr::get(outFT, dv));
  }

  if (auto outIT = mlir::dyn_cast<IntegerType>(outET)) {
    IntegerType clampIT =
        mlir::isa<IntegerType>(outET) ? mlir::cast<IntegerType>(outET) : outIT;

    int64_t iv = static_cast<int64_t>(std::llround(d));
    const unsigned bw = clampIT.getWidth();
    const bool isSigned = clampIT.isSigned();

    const int64_t minV = isSigned ? (-(int64_t(1) << (bw - 1))) : 0;
    const int64_t maxV =
        isSigned ? ((int64_t(1) << (bw - 1)) - 1) : ((int64_t(1) << bw) - 1);
    iv = std::min<int64_t>(std::max<int64_t>(iv, minV), maxV);

    if (outIT.isSigned()) {
      return DenseElementsAttr::get(ranked, IntegerAttr::get(outIT, iv));
    } else {
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

  auto qOp = dqOp.getX().template getDefiningOp<ONNXQuantizeLinearOp>();
  if (!qOp)
    return failure();

  if (qOp.getAxis() != dqOp.getAxis())
    return failure();
  if (qOp.getBlockSize() != dqOp.getBlockSize())
    return failure();

  auto zpQ = getElementAttributeFromONNXValue(qOp.getYZeroPoint());
  auto zpDQ = getElementAttributeFromONNXValue(dqOp.getXZeroPoint());
  if (!zpQ || !zpDQ || zpQ != zpDQ)
    return failure();

  auto sQ = getElementAttributeFromONNXValue(qOp.getYScale());
  auto sDQ = getElementAttributeFromONNXValue(dqOp.getXScale());
  if (!sQ || !sDQ || sQ != sDQ)
    return failure();

  auto qInTypeOp = qOp.getX().getType();
  auto dqOutTypeOp = dqOp.getResult().getType();
  auto qInT = mlir::dyn_cast<TensorType>(qInTypeOp);
  auto dqOutT = mlir::dyn_cast<TensorType>(dqOutTypeOp);
  if (!qInT || !dqOutT)
    return failure();
  if (dqOutT.getElementType() != qInT.getElementType())
    return failure();

  if (!doRewrite)
    return success();

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
    SmallVector<ONNXDequantizeLinearOp, 4> matching_dequantize_nodes;

    for (Operation *user : qOp.getY().getUsers()) {
      if (auto tailDQ = llvm::dyn_cast<ONNXDequantizeLinearOp>(user)) {
        if (succeeded(
                tryRemoveQThenDQChain(rewriter, tailDQ, /*doRewrite*/ false))) {
          matching_dequantize_nodes.push_back(tailDQ);
        }
      }
    }

    if (matching_dequantize_nodes.empty())
      return false;

    for (auto dqOp : matching_dequantize_nodes) {
      (void)tryRemoveQThenDQChain(rewriter, dqOp, /*doRewrite*/ true);
    }

    if (qOp->use_empty()) {
      rewriter.eraseOp(qOp);
    }

    return true;

  } else {
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

    mlir::Operation *dstNode =
        nullptr; // DQ when folding into DQ, or Q when folding into Q
    mlir::Operation *srcNode = nullptr;

    mlir::Value dstScaleValue;     // Scale Value for reuse
    mlir::Value dstZeroPointValue; // Zero point Value for reuse
    double dstScale = 0.0;
    int64_t dstZeroPoint = 0;

    double newScale = 0.0;
    int64_t newZp = 0;

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

    if (constantIsFirstOperand &&
        (llvm::isa<ONNXSubOp>(binaryOp) || llvm::isa<ONNXDivOp>(binaryOp))) {
      return rewriter.notifyMatchFailure(binaryOp,
          "Qdq initializer: Div and Sub are not supported when "
          "weight is the first input");
    }

    {
      if (isConstantFromQDQChain) {
        auto scalar_value_opt = getScalarTensorValue<double>(constantSourceOp);
        if (!scalar_value_opt) {
          return rewriter.notifyMatchFailure(constantSourceOp,
              " must be a scalar value or a list of same value");
        }
        state.kValue = *scalar_value_opt;
      } else {
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

    if (auto dqOp = lhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
      if (auto constOp = rhs.getDefiningOp<ONNXConstantOp>()) {
        state.dequantActivationOfBinOp = dqOp;
        constantOp = constOp;
      }
    } else if (auto dqOp = rhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
      if (auto constOp = lhs.getDefiningOp<ONNXConstantOp>()) {
        if (llvm::isa<ONNXSubOp>(binaryOp) || llvm::isa<ONNXDivOp>(binaryOp)) {
          return rewriter.notifyMatchFailure(binaryOp,
              "non-qdq initializer: Div and Sub are not supported "
              "when weight is the first input");
        }
        state.dequantActivationOfBinOp = dqOp;
        constantOp = constOp;
      }
    }

    if (state.dequantActivationOfBinOp && constantOp) {
      auto kValueOpt = getScalarTensorValue<double>(constantOp);
      if (!kValueOpt) {
        return rewriter.notifyMatchFailure(
            constantOp, " must be a scalar value or a list of same value");
      }
      state.kValue = kValueOpt.value();
      return success();
    }

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

    Value scaleVal = state.dstScaleValue;
    Value zpVal = state.dstZeroPointValue;

    auto scaleType = mlir::dyn_cast<ShapedType>(scaleVal.getType());
    auto zpType = mlir::dyn_cast<ShapedType>(zpVal.getType());

    if (!scaleType || !zpType)
      return rewriter.notifyMatchFailure(
          state.dstNode, "scale or zero point is not a shaped type");

    Type scaleElemType = scaleType.getElementType();
    Type zpElemType = zpType.getElementType();

    if (auto zpIntType = mlir::dyn_cast<IntegerType>(zpElemType)) {
      int64_t zpMin, zpMax;
      unsigned bitWidth = zpIntType.getWidth();

      if (bitWidth == 4) {
        if (zpIntType.isUnsigned()) {
          zpMin = 0;
          zpMax = 15;
        } else {
          zpMin = -8;
          zpMax = 7;
        }
      } else {
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

    if (auto scaleFloatType = mlir::dyn_cast<FloatType>(scaleElemType)) {
      double scaleMin, scaleMax;

      if (scaleFloatType.isF16()) {
        scaleMin = -65504.0;
        scaleMax = 65504.0;
      } else if (scaleFloatType.isBF16()) {
        scaleMin = -std::numeric_limits<float>::max();
        scaleMax = std::numeric_limits<float>::max();
      } else if (scaleFloatType.isF32()) {
        scaleMin = -std::numeric_limits<float>::max();
        scaleMax = std::numeric_limits<float>::max();
      } else if (scaleFloatType.isF64()) {
        scaleMin = -std::numeric_limits<double>::max();
        scaleMax = std::numeric_limits<double>::max();
      } else {
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

    auto hasBranchOnValue = [](mlir::Value v) {
      llvm::SmallPtrSet<mlir::Operation *, 8> uniq;
      for (auto *u : v.getUsers())
        uniq.insert(u);
      return uniq.size() > 1;
    };

    const bool branch_after = q && hasBranchOnValue(dq.getY());

    const bool branch_before = hasBranchOnValue(dq.getX());
    const bool branch_on_dequant_activation = branch_after || branch_before;

    if (!removableQDQ || branch_on_dequant_activation) {
      state.dstNode = dq.getOperation();
      state.srcNode = state.quantOutputOfBinOp.getOperation();

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

    auto qOut = state.quantOutputOfBinOp;
    if (!qOut)
      return rewriter.notifyMatchFailure(
          op, "expected a unique Quantize user of the binary result");

    state.dstNode = qOut.getOperation();
    state.srcNode = state.dequantActivationOfBinOp.getOperation();

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

    if (!op->hasOneUse()) {
      return rewriter.notifyMatchFailure(op, "pattern requires a single user");
    }
    auto quantOutputOfBinOp =
        mlir::dyn_cast<ONNXQuantizeLinearOp>(*op->user_begin());
    if (!quantOutputOfBinOp) {
      return rewriter.notifyMatchFailure(
          op, "expected user to be an ONNXQuantizeLinearOp");
    }

    MatchState state;

    if (failed(match_binary_op(rewriter, state, op))) {
      return rewriter.notifyMatchFailure(op,
          " does not match to critieria to remove binary. Remove binary op "
          "only if one of the dequantize linear input "
          "has const scalar value ");
    }

    if (failed(findDestinationNode(rewriter, state, op))) {
      return failure();
    }

    if (failed(check_needed_values(rewriter, state, op))) {
      return failure();
    }

    if (!compute_new_scale_and_zp_values(state)) {
      return failure();
    }

    if (failed(checkNewQDQParameterFits(rewriter, state))) {
      return failure();
    }

    {
      if (!state.dstNode)
        return rewriter.notifyMatchFailure(op, "dstNode not set");

      if constexpr (std::is_same_v<BinOp, ONNXAddOp> ||
                    std::is_same_v<BinOp, ONNXSubOp>) {
        updateInitializer(rewriter, state.dstNode, state.dstZeroPointValue,
            static_cast<double>(state.newZp));
      } else if constexpr (std::is_same_v<BinOp, ONNXMulOp> ||
                           std::is_same_v<BinOp, ONNXDivOp>) {
        updateInitializer(
            rewriter, state.dstNode, state.dstScaleValue, state.newScale);
      }
    }

    rewriter.replaceOp(op, state.dequantActivationOfBinOp.getResult());

    ONNXQuantizeLinearOp chainStartQ = nullptr;

    if (llvm::isa<ONNXDequantizeLinearOp>(state.dstNode)) {
      chainStartQ = state.quantOutputOfBinOp;
    } else if (llvm::isa<ONNXQuantizeLinearOp>(state.dstNode)) {
      if (auto dqAct = state.dequantActivationOfBinOp) {
        chainStartQ =
            dqAct.getX().template getDefiningOp<ONNXQuantizeLinearOp>();
      }
    }
    if (chainStartQ) {
      (void)Remove_Q_Plus_DQ(rewriter, chainStartQ, /*doRewrite=*/true);
    }

    return success();
  }
};

// ---------------------------------------------------------------------------
// Shared range-check helper for new scale / zero-point values.
// Returns success() when both values fit their respective element types.
// ---------------------------------------------------------------------------
static LogicalResult checkNewParamsFit(PatternRewriter &rewriter,
    Operation *contextOp, Value scaleVal, Value zpVal, double newScale,
    int64_t newZp) {
  auto scaleType = mlir::dyn_cast<ShapedType>(scaleVal.getType());
  auto zpType = mlir::dyn_cast<ShapedType>(zpVal.getType());
  if (!scaleType || !zpType)
    return rewriter.notifyMatchFailure(
        contextOp, "scale or zero point is not a shaped type");

  if (auto zpIntType = mlir::dyn_cast<IntegerType>(zpType.getElementType())) {
    int64_t zpMin, zpMax;
    unsigned bw = zpIntType.getWidth();
    if (bw == 4) {
      zpMin = zpIntType.isUnsigned() ? 0 : -8;
      zpMax = zpIntType.isUnsigned() ? 15 : 7;
    } else if (zpIntType.isUnsigned()) {
      zpMin = 0;
      zpMax = (bw == 64) ? INT64_MAX : ((int64_t(1) << bw) - 1);
    } else {
      zpMin = (bw == 64) ? INT64_MIN : (-(int64_t(1) << (bw - 1)));
      zpMax = (bw == 64) ? INT64_MAX : ((int64_t(1) << (bw - 1)) - 1);
    }
    if (newZp < zpMin || newZp > zpMax)
      return rewriter.notifyMatchFailure(contextOp,
          ("new zero point " + std::to_string(newZp) + " out of [" +
              std::to_string(zpMin) + "," + std::to_string(zpMax) + "]")
              .c_str());
  }

  if (auto scaleFT = mlir::dyn_cast<FloatType>(scaleType.getElementType())) {
    double scaleMax;
    if (scaleFT.isF16())
      scaleMax = 65504.0;
    else if (scaleFT.isBF16() || scaleFT.isF32())
      scaleMax = std::numeric_limits<float>::max();
    else if (scaleFT.isF64())
      scaleMax = std::numeric_limits<double>::max();
    else
      return rewriter.notifyMatchFailure(
          contextOp, "unsupported float type for scale");
    if (newScale < -scaleMax || newScale > scaleMax)
      return rewriter.notifyMatchFailure(contextOp,
          ("new scale " + std::to_string(newScale) + " out of range").c_str());
  }
  return success();
}

// ---------------------------------------------------------------------------
// Case 1: activation -> BinaryOp(constant) -> Q
//
// No DequantizeLinear feeds the activation side.  The constant is folded into
// Q's y_scale / y_zero_point so the BinaryOp can be removed.
//
//   Mul:  q = round(x*k / s) + z  =>  s' = s/k, z' = z
//   Div:  q = round(x/k / s) + z  =>  s' = s*k, z' = z
//   Add:  q = round((x+k)/s) + z  =>  s' = s,   z' = z + k/s
//   Sub:  q = round((x-k)/s) + z  =>  s' = s,   z' = z - k/s
// ---------------------------------------------------------------------------
template <typename BinOp>
struct FoldBinaryIntoQ : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      BinOp op, PatternRewriter &rewriter) const override {

    if (!op->hasOneUse())
      return rewriter.notifyMatchFailure(op, "BinaryOp must have one user");

    auto qOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(*op->user_begin());
    if (!qOp)
      return rewriter.notifyMatchFailure(
          op, "single user must be QuantizeLinear");

    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    ONNXConstantOp constantOp = nullptr;
    Value activation;

    if (auto cOp = rhs.getDefiningOp<ONNXConstantOp>()) {
      constantOp = cOp;
      activation = lhs;
    } else if (auto cOp = lhs.getDefiningOp<ONNXConstantOp>()) {
      if constexpr (std::is_same_v<BinOp, ONNXSubOp> ||
                    std::is_same_v<BinOp, ONNXDivOp>)
        return rewriter.notifyMatchFailure(
            op, "Sub/Div not supported when constant is first operand");
      constantOp = cOp;
      activation = rhs;
    }

    if (!constantOp)
      return rewriter.notifyMatchFailure(
          op, "one operand must be a constant scalar");

    // Skip if activation comes from DQ (existing FoldBinaryThroughQDQ handles
    // the full DQ -> BinaryOp -> Q sandwich).
    if (activation.getDefiningOp<ONNXDequantizeLinearOp>())
      return rewriter.notifyMatchFailure(
          op, "activation from DQ handled by FoldBinaryThroughQDQ");

    auto kOpt = getScalarTensorValue<double>(constantOp);
    if (!kOpt)
      return rewriter.notifyMatchFailure(
          op, "constant must be a scalar or all-same tensor");
    double k = *kOpt;

    // Read Q's current y_scale / y_zero_point.
    Value scaleVal = qOp.getYScale();
    Value zpVal = qOp.getYZeroPoint();
    auto scaleOpt = getScalarTensorValueFromVal<double>(scaleVal);
    auto zpOpt = getScalarTensorValueFromVal<int64_t>(zpVal);
    if (!scaleOpt || !zpOpt)
      return rewriter.notifyMatchFailure(
          op, "Q y_scale/y_zero_point must be scalar");
    double scale = *scaleOpt;
    int64_t zp = *zpOpt;

    // Compute new scale / zero-point (folding into Q).
    double newScale = scale;
    double newZpF = static_cast<double>(zp);

    if constexpr (std::is_same_v<BinOp, ONNXAddOp>) {
      if (scale == 0.0)
        return rewriter.notifyMatchFailure(op, "scale is zero (Add)");
      newZpF += (k / scale);
    } else if constexpr (std::is_same_v<BinOp, ONNXSubOp>) {
      if (scale == 0.0)
        return rewriter.notifyMatchFailure(op, "scale is zero (Sub)");
      newZpF -= (k / scale);
    } else if constexpr (std::is_same_v<BinOp, ONNXMulOp>) {
      if (k == 0.0)
        return rewriter.notifyMatchFailure(op, "k is zero (Mul)");
      newScale /= k;
    } else if constexpr (std::is_same_v<BinOp, ONNXDivOp>) {
      newScale *= k;
    }

    int64_t newZp = (newZpF >= 0.0) ? static_cast<int64_t>(std::floor(newZpF))
                                    : static_cast<int64_t>(std::ceil(newZpF));

    if (failed(
            checkNewParamsFit(rewriter, op, scaleVal, zpVal, newScale, newZp)))
      return failure();

    // Update Q's parameters.
    if constexpr (std::is_same_v<BinOp, ONNXAddOp> ||
                  std::is_same_v<BinOp, ONNXSubOp>) {
      updateInitializer(rewriter, qOp, zpVal, static_cast<double>(newZp));
    } else {
      updateInitializer(rewriter, qOp, scaleVal, newScale);
    }

    // Remove the BinaryOp: Q now takes the activation directly.
    rewriter.replaceOp(op, activation);
    return success();
  }
};

// ---------------------------------------------------------------------------
// Case 2: DQ -> BinaryOp(constant) -> consumer  (consumer is NOT Q)
//
// The constant is folded into DQ's x_scale / x_zero_point so the BinaryOp
// can be removed.  DQ must be single-use (only feeds this BinaryOp) so that
// modifying its params does not affect other consumers.
//
//   x_float = (q - zp) * s
//
//   Mul:  x_float * k = (q - zp) * (s*k)   =>  s' = s*k
//   Div:  x_float / k = (q - zp) * (s/k)   =>  s' = s/k
//   Add:  x_float + k = (q - (zp - k/s))*s  =>  z' = zp - k/s
//   Sub:  x_float - k = (q - (zp + k/s))*s  =>  z' = zp + k/s
// ---------------------------------------------------------------------------
template <typename BinOp>
struct FoldBinaryIntoDQ : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      BinOp op, PatternRewriter &rewriter) const override {

    if (!op->hasOneUse())
      return rewriter.notifyMatchFailure(op, "BinaryOp must have one user");

    // If the single user is Q, the full DQ->BinaryOp->Q pattern handles it.
    if (mlir::dyn_cast<ONNXQuantizeLinearOp>(*op->user_begin()))
      return rewriter.notifyMatchFailure(
          op, "user is Q; handled by FoldBinaryThroughQDQ");

    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    ONNXDequantizeLinearOp dqOp = nullptr;
    ONNXConstantOp constantOp = nullptr;

    // Try: lhs=DQ, rhs=constant
    if (auto dq = lhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
      if (auto cOp = rhs.getDefiningOp<ONNXConstantOp>()) {
        dqOp = dq;
        constantOp = cOp;
      }
    }
    // Try: lhs=constant, rhs=DQ (only for commutative ops)
    if (!dqOp) {
      if (auto dq = rhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
        if (auto cOp = lhs.getDefiningOp<ONNXConstantOp>()) {
          if constexpr (std::is_same_v<BinOp, ONNXSubOp> ||
                        std::is_same_v<BinOp, ONNXDivOp>)
            return rewriter.notifyMatchFailure(
                op, "Sub/Div not supported when constant is first operand");
          dqOp = dq;
          constantOp = cOp;
        }
      }
    }

    if (!dqOp || !constantOp)
      return rewriter.notifyMatchFailure(
          op, "need one DQ input and one constant scalar input");

    // DQ must be single-use so modifying its params is safe.
    if (!dqOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "DQ has multiple users; unsafe to modify in place");

    auto kOpt = getScalarTensorValue<double>(constantOp);
    if (!kOpt)
      return rewriter.notifyMatchFailure(
          op, "constant must be a scalar or all-same tensor");
    double k = *kOpt;

    // Read DQ's current x_scale / x_zero_point.
    Value scaleVal = dqOp.getXScale();
    Value zpVal = dqOp.getXZeroPoint();
    auto scaleOpt = getScalarTensorValueFromVal<double>(scaleVal);
    auto zpOpt = getScalarTensorValueFromVal<int64_t>(zpVal);
    if (!scaleOpt || !zpOpt)
      return rewriter.notifyMatchFailure(
          op, "DQ x_scale/x_zero_point must be scalar");
    double scale = *scaleOpt;
    int64_t zp = *zpOpt;

    // Compute new scale / zero-point (folding into DQ).
    double newScale = scale;
    double newZpF = static_cast<double>(zp);

    if constexpr (std::is_same_v<BinOp, ONNXAddOp>) {
      if (scale == 0.0)
        return rewriter.notifyMatchFailure(op, "scale is zero (Add)");
      newZpF -= (k / scale);
    } else if constexpr (std::is_same_v<BinOp, ONNXSubOp>) {
      if (scale == 0.0)
        return rewriter.notifyMatchFailure(op, "scale is zero (Sub)");
      newZpF += (k / scale);
    } else if constexpr (std::is_same_v<BinOp, ONNXMulOp>) {
      newScale *= k;
    } else if constexpr (std::is_same_v<BinOp, ONNXDivOp>) {
      if (k == 0.0)
        return rewriter.notifyMatchFailure(op, "k is zero (Div)");
      newScale /= k;
    }

    int64_t newZp = (newZpF >= 0.0) ? static_cast<int64_t>(std::floor(newZpF))
                                    : static_cast<int64_t>(std::ceil(newZpF));

    if (failed(
            checkNewParamsFit(rewriter, op, scaleVal, zpVal, newScale, newZp)))
      return failure();

    // Update DQ's parameters.
    if constexpr (std::is_same_v<BinOp, ONNXAddOp> ||
                  std::is_same_v<BinOp, ONNXSubOp>) {
      updateInitializer(rewriter, dqOp, zpVal, static_cast<double>(newZp));
    } else {
      updateInitializer(rewriter, dqOp, scaleVal, newScale);
    }

    // Remove the BinaryOp: consumers now take DQ output directly.
    rewriter.replaceOp(op, dqOp.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct DQBinaryQOptPass
    : public PassWrapper<DQBinaryQOptPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "dq-binary-q-opt"; }
  StringRef getDescription() const override {
    return "Fold scalar binary ops (Add/Sub/Mul/Div) between "
           "DequantizeLinear and QuantizeLinear into quantization parameters";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // Existing: DQ -> BinaryOp(constant) -> Q
    // patterns.add<FoldBinaryThroughQDQ<ONNXDivOp>>(context);
    // patterns.add<FoldBinaryThroughQDQ<ONNXSubOp>>(context);
    // patterns.add<FoldBinaryThroughQDQ<ONNXMulOp>>(context);
    // patterns.add<FoldBinaryThroughQDQ<ONNXAddOp>>(context);

    // Case 1: activation -> BinaryOp(constant) -> Q  (no DQ before)
    // patterns.add<FoldBinaryIntoQ<ONNXAddOp>>(context);
    // patterns.add<FoldBinaryIntoQ<ONNXSubOp>>(context);
    patterns.add<FoldBinaryIntoQ<ONNXMulOp>>(context);
    // patterns.add<FoldBinaryIntoQ<ONNXDivOp>>(context);

    // Case 2: DQ -> BinaryOp(constant) -> consumer  (no Q after)
    // patterns.add<FoldBinaryIntoDQ<ONNXAddOp>>(context);
    // patterns.add<FoldBinaryIntoDQ<ONNXSubOp>>(context);
    patterns.add<FoldBinaryIntoDQ<ONNXMulOp>>(context);
    // patterns.add<FoldBinaryIntoDQ<ONNXDivOp>>(context);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createDQBinaryQOptPass() {
  return std::make_unique<DQBinaryQOptPass>();
}

} // namespace onnx_mlir
