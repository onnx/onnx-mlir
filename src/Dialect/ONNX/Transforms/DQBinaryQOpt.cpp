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
#include <cmath> // For std::llround
#include <optional>
#include <variant>

using namespace mlir;
using namespace onnx_mlir;

namespace {

static ElementsAttr getElementAttributeFromConstant(Value val) {
  if (!val)
    return nullptr;
  if (auto constOp = val.getDefiningOp<ONNXConstantOp>())
    return mlir::dyn_cast<ElementsAttr>(constOp.getValueAttr());
  return nullptr;
}

// Equivalent to Python's NoMatch exception: here Nullopt indicates failure.
template <typename T>
std::optional<T> get_scalar_tensor_value(ONNXConstantOp constOp) {
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(constOp.getValueAttr());
  if (!elementsAttr)
    return std::nullopt;

  Type elementType = elementsAttr.getElementType();

  // Fast path: splat
  if (elementsAttr.isSplat()) {
    if (elementType.isa<FloatType>()) {
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
        APFloat splatValue = elementsAttr.getSplatValue<APFloat>();
        return static_cast<T>(splatValue.convertToDouble());
      }
    }
    if (auto intType = elementType.dyn_cast<IntegerType>()) {
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
  auto shapedTy = elementsAttr.getType().dyn_cast<ShapedType>();
  if (!shapedTy || !shapedTy.hasStaticShape())
    return std::nullopt;

  // Case: rank 0 → scalar element directly
  if (shapedTy.getRank() == 0) {
    auto firstAttr = *elementsAttr.getValues<Attribute>().begin();
    if (auto fAttr = firstAttr.dyn_cast<FloatAttr>()) {
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
        return static_cast<T>(fAttr.getValueAsDouble());
    }
    if (auto iAttr = firstAttr.dyn_cast<IntegerAttr>()) {
      if constexpr (std::is_integral_v<T>)
        return static_cast<T>(iAttr.getInt()); // signed ok
    }
    return std::nullopt;
  }

  // Case: rank >= 1 → flatten & check all the same
  std::set<double> flattenedFP;
  std::set<int64_t> flattenedInt;

  if (elementType.isa<FloatType>()) {
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
      for (auto a : elementsAttr.getValues<FloatAttr>())
        flattenedFP.insert(a.getValueAsDouble());
      if (flattenedFP.size() == 1)
        return static_cast<T>(*flattenedFP.begin());
    }
  } else if (auto intType = elementType.dyn_cast<IntegerType>()) {
    if constexpr (std::is_integral_v<T>) {
      for (auto a : elementsAttr.getValues<IntegerAttr>())
        flattenedInt.insert(intType.isUnsigned() ? a.getUInt() : a.getInt());
      if (flattenedInt.size() == 1)
        return static_cast<T>(*flattenedInt.begin());
    }
  }

  return std::nullopt; // mismatch or more than one unique value
}

template <typename T>
std::optional<T> get_scalar_tensor_value_from_val(Value value) {
  if (!value) {
    return std::nullopt;
  }
  auto constOp = value.getDefiningOp<ONNXConstantOp>();
  if (!constOp) {
    return std::nullopt;
  }
  return get_scalar_tensor_value<T>(constOp);
}

static mlir::DenseElementsAttr makeScalarDEA(
    mlir::ShapedType likeTy, double d, mlir::Type clampElemTy) {
  using namespace mlir;

  auto ranked = likeTy.dyn_cast<RankedTensorType>();
  if (!ranked || !ranked.hasStaticShape() || ranked.getNumElements() != 1)
    return {};

  Type outET = ranked.getElementType();
  Type useET = clampElemTy ? clampElemTy : outET;

  // If target is float, just create a float attr with outET semantics.
  if (auto outFT = outET.dyn_cast<FloatType>()) {
    // Round in the semantics of useET if it's float; otherwise just use d.
    double dv = d;
    if (auto useFT = useET.dyn_cast<FloatType>()) {
      // Convert through APFloat with 'useET' semantics, then to double.
      llvm::APFloat ap(d);
      bool loses = false;
      ap.convert(useFT.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
          &loses);
      dv = ap.convertToDouble();
    }
    return DenseElementsAttr::get(ranked, FloatAttr::get(outFT, dv));
  }

  // If target is integer, round+clamp as per 'useET' (if integer), then emit as
  // outET.
  if (auto outIT = outET.dyn_cast<IntegerType>()) {
    // Decide signedness/width for clamping from useET if it's integer, else
    // from outET.
    IntegerType clampIT =
        useET.isa<IntegerType>() ? useET.cast<IntegerType>() : outIT;

    int64_t iv = static_cast<int64_t>(std::llround(d));
    const unsigned bw = clampIT.getWidth();
    const bool isSigned = clampIT.isSigned();

    const int64_t minV = isSigned ? (-(int64_t(1) << (bw - 1))) : 0;
    const int64_t maxV =
        isSigned ? ((int64_t(1) << (bw - 1)) - 1) : ((int64_t(1) << bw) - 1);
    iv = std::min<int64_t>(std::max<int64_t>(iv, minV), maxV);

    // Now re-materialize as the *output* ET (which may differ in width/sign).
    // This guarantees the result type matches `likeTy`.
    if (auto outSigned = outIT.isSigned()) {
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
    mlir::Operation *targetOp, mlir::Value oldInit, double newScalar,
    mlir::Type clampElemTy) {
  using namespace mlir;

  if (!targetOp || !oldInit)
    return;

  auto oldCst = oldInit.getDefiningOp<ONNXConstantOp>();
  if (!oldCst)
    return;

  auto likeTy = oldInit.getType().dyn_cast<ShapedType>();
  llvm::outs() << "likeTy = " << likeTy << " clampElemTy = " << clampElemTy
               << " newScalar = " << newScalar << "\n";
  if (!likeTy || !likeTy.hasStaticShape() || likeTy.getNumElements() != 1)
    return;

  DenseElementsAttr payload = makeScalarDEA(likeTy, newScalar, clampElemTy);
  if (!payload)
    return;

  // Single-use-by-target check.
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
      // Keep constant canonical:
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

  // Multi-use: clone a fresh constant with same result type as oldInit.
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

  // Replace exactly the matching operand.
  for (unsigned i = 0, e = targetOp->getNumOperands(); i < e; ++i) {
    if (targetOp->getOperand(i) == oldInit) {
      targetOp->setOperand(i, newCst.getOutput());
      break;
    }
  }
}

static LogicalResult tryRemoveQThenDQChain(
    mlir::PatternRewriter &rewriter, mlir::ONNXDequantizeLinearOp dqOp) {
  using namespace mlir;

  // Match Q -> DQ
  auto qOp = dqOp.getX().template getDefiningOp<ONNXQuantizeLinearOp>();
  if (!qOp) {
    return failure();
  }

  // 1) Axis / block_size must match
  if (qOp.getAxis() != dqOp.getAxis()) {
    return failure();
  }
  if (qOp.getBlockSize() != dqOp.getBlockSize()) {
    return failure();
  }

  // 2) Zero-points must match scalars/splats
  auto zpQ = getElementAttributeFromConstant(qOp.getYZeroPoint());
  auto zpDQ = getElementAttributeFromConstant(dqOp.getXZeroPoint());
  if (!zpQ || !zpDQ || zpQ != zpDQ) {
    return failure();
  }

  // 3) Scales must match scalars/splats
  auto sQ = getElementAttributeFromConstant(qOp.getYScale());
  auto sDQ = getElementAttributeFromConstant(dqOp.getXScale());
  if (!sQ || !sDQ || sQ != sDQ) {
    return failure();
  }

  // 4) Data type consistency: input of Q and output of DQ must have same elem
  // type.
  auto qInTypeOp = qOp.getX().getType();
  auto dqOutTypeOp = dqOp.getResult().getType();

  if (auto qInTensorType = qInTypeOp.dyn_cast<TensorType>()) {
    if (auto dqOutTensorType = dqOutTypeOp.dyn_cast<TensorType>()) {
      if (dqOutTensorType.getElementType() != qInTensorType.getElementType()) {
        return failure();
      }
    } else {
      return failure();
    }
  } else {
    return failure();
  }

  // Replace DQ with Q's float input; erase Q if it becomes dead.
  rewriter.replaceOp(dqOp, qOp.getX());
  if (qOp->use_empty()) {
    rewriter.eraseOp(qOp);
  }

  return success();
}

template <typename BinOp>
struct FoldBinaryThroughQDQ : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

private:
  struct MatchState {
    ONNXDequantizeLinearOp dequantActivationOp = nullptr;
    mlir::Type ScaleDtype;
    mlir::Type zeroPointDtype;
    double kValue = 0.0;
    double dstScale = 0.0;
    int64_t dstZeroPoint = 0;
    double newScale = 0.0;
    int64_t newZp = 0;
  };

  LogicalResult match_qdq(MatchState &state, ONNXDequantizeLinearOp dq1,
      ONNXDequantizeLinearOp dq2) const {

    ONNXDequantizeLinearOp constantDqOp = nullptr;
    ONNXConstantOp constantSourceOp = nullptr;

    // Case 1: Direct ConstantOp as input to the DQ.
    if (auto constOp = dq1.getX().getDefiningOp<ONNXConstantOp>()) {
      constantDqOp = dq1;
      state.dequantActivationOp = dq2;
      constantSourceOp = constOp;
    } else if (auto constOp = dq2.getX().getDefiningOp<ONNXConstantOp>()) {
      constantDqOp = dq2;
      state.dequantActivationOp = dq1;
      constantSourceOp = constOp;
    }
    // Case 2: The input to the DQ op comes from a chain whose input is a
    // constant.
    else if (auto intermediateOp = dq1.getX().getDefiningOp()) {
      if (auto constOp =
              intermediateOp->getOperand(0).getDefiningOp<ONNXConstantOp>()) {
        constantDqOp = dq1;
        state.dequantActivationOp = dq2;
        constantSourceOp = constOp;
      }
    } else if (auto intermediateOp = dq2.getX().getDefiningOp()) {
      if (auto constOp =
              intermediateOp->getOperand(0).getDefiningOp<ONNXConstantOp>()) {
        constantDqOp = dq2;
        state.dequantActivationOp = dq1;
        constantSourceOp = constOp;
      }
    }

    if (!constantDqOp || !constantSourceOp || !state.dequantActivationOp) {
      return failure();
    }

    // Find kvalue and store scale_dtype and zeroPointDtype
    {
      auto scalar_value_opt =
          get_scalar_tensor_value<int64_t>(constantSourceOp);
      if (!scalar_value_opt) {
        return failure();
      }
      Value scaleVal = constantDqOp.getXScale();
      Value zpVal = constantDqOp.getXZeroPoint();
      auto scale_value_opt = get_scalar_tensor_value_from_val<double>(scaleVal);
      auto zp_value_opt = get_scalar_tensor_value_from_val<int64_t>(zpVal);
      if (!scale_value_opt || !zp_value_opt) {
        return failure();
      }
      // Calculate and store kValue.
      state.kValue = (*scalar_value_opt - *zp_value_opt) * *scale_value_opt;

      // # store dtype for creating new initializers with the same dtype
      state.ScaleDtype =
          mlir::cast<ShapedType>(scaleVal.getType()).getElementType();
      state.zeroPointDtype =
          mlir::cast<ShapedType>(zpVal.getType()).getElementType();
    }
    return success();
  }

  LogicalResult match_binary_op(MatchState &state, BinOp binaryOp) const {
    ONNXConstantOp constantOp = nullptr;

    Value lhs = binaryOp.getOperand(0);
    Value rhs = binaryOp.getOperand(1);

    // -------- Case A: lhs is DQ, rhs is Constant --------
    if (auto dqOp = lhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
      if (auto constOp = rhs.getDefiningOp<ONNXConstantOp>()) {
        state.dequantActivationOp = dqOp;
        constantOp = constOp;
      }
    }
    // -------- Case A reversed --------
    else if (auto dqOp = rhs.getDefiningOp<ONNXDequantizeLinearOp>()) {
      if (auto constOp = lhs.getDefiningOp<ONNXConstantOp>()) {
        state.dequantActivationOp = dqOp;
        constantOp = constOp;
      }
    }

    // -------- Fill state values for Case A and Case A reversed --------
    if (state.dequantActivationOp && constantOp) {
      auto kValueOpt = get_scalar_tensor_value<double>(constantOp);
      if (!kValueOpt) {
        return failure();
      }
      state.kValue = kValueOpt.value();
      return success();
    }

    // -------- Case B: both inputs are DQ --------
    auto dqOp1 = lhs.getDefiningOp<ONNXDequantizeLinearOp>();
    auto dqOp2 = rhs.getDefiningOp<ONNXDequantizeLinearOp>();

    if (dqOp1 && dqOp2) {
      if (failed(match_qdq(state, dqOp1, dqOp2)))
        return failure();
      return success();
    }
    return failure();
  }

  LogicalResult check_needed_values(
      const MatchState &state, Operation *binaryOp) const {
    if (state.kValue == 0.0) {
      if (isa<ONNXDivOp>(binaryOp)) {
        return failure();
      }
    }
    if (state.dstScale == 0.0) {
      if (isa<ONNXAddOp, ONNXSubOp>(binaryOp)) {
        return failure();
      }
    }
    return success();
  }

  static bool compute_new_scale_and_zp_values(
      MatchState &state, Operation *binaryOp) {
    double newScale = state.dstScale;
    double newZpFloat = static_cast<double>(state.dstZeroPoint);
    const double kVal = state.kValue;

    if (isa<ONNXAddOp>(binaryOp)) {
      newZpFloat -= (kVal / newScale);

    } else if (isa<ONNXSubOp>(binaryOp)) {
      newZpFloat += (kVal / newScale);

    } else if (isa<ONNXMulOp>(binaryOp)) {
      newScale *= kVal;

    } else if (isa<ONNXDivOp>(binaryOp)) {
      newScale /= kVal;

    } else {
      return false;
    }

    int64_t newZp = static_cast<int64_t>(std::llround(newZpFloat));
    state.newScale = newScale;
    state.newZp = newZp;

    return true;
  }

public:
  LogicalResult matchAndRewrite(
      BinOp op, PatternRewriter &rewriter) const override {

    // STEP 1: Match begin: Assuming only one user
    auto quantOutputOp = dyn_cast<ONNXQuantizeLinearOp>(*op->user_begin());
    if (!quantOutputOp) {
      return failure();
    }

    // Instantiate the state struct
    MatchState state;

    // STEP 2
    if (failed(match_binary_op(state, op))) {
      return failure();
    }

    // Store the value of the scale and zero point of the destination node
    {
      Value scaleVal = state.dequantActivationOp.getXScale();
      Value zpVal = state.dequantActivationOp.getXZeroPoint();
      auto scale_value_opt = get_scalar_tensor_value_from_val<double>(scaleVal);
      auto zp_value_opt = get_scalar_tensor_value_from_val<int64_t>(zpVal);
      if (!scale_value_opt || !zp_value_opt) {
        return failure();
      }
      state.dstScale = scale_value_opt.value();
      state.dstZeroPoint = zp_value_opt.value();
    }

    // STEP 3
    if (failed(check_needed_values(state, op))) {
      return failure();
    }

    // STEP 4 -Modify
    if (!compute_new_scale_and_zp_values(state, op)) {
      return failure();
    }

    // STEP 5: call initializer based on the binary op
    ONNXDequantizeLinearOp dqAct = state.dequantActivationOp;
    if constexpr (std::is_same_v<BinOp, ONNXAddOp> ||
                  std::is_same_v<BinOp, ONNXSubOp>) {
      Value zpVal = dqAct.getXZeroPoint();
      updateInitializer(rewriter, dqAct.getOperation(), zpVal,
          static_cast<double>(state.newZp), state.zeroPointDtype);

    } else if constexpr (std::is_same_v<BinOp, ONNXMulOp> ||
                         std::is_same_v<BinOp, ONNXDivOp>) {
      Value scaleVal = dqAct.getXScale();
      updateInitializer(rewriter, dqAct.getOperation(), scaleVal,
          state.newScale, state.ScaleDtype);
    }

    // STEP 6: Remove binary op
    rewriter.replaceOp(op, dqAct.getResult());

    // STEP 7: Remove Q->DQ chain
    for (Operation *user : quantOutputOp.getY().getUsers()) {
      if (auto tailDQ = llvm::dyn_cast<ONNXDequantizeLinearOp>(user)) {
        (void)tryRemoveQThenDQChain(rewriter, tailDQ);
      }
    }

    return success();
  }
};

struct FoldDQBinaryQPass
    : public PassWrapper<FoldDQBinaryQPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldDQBinaryQPass)

  StringRef getArgument() const final { return "dq-binary-q-opt-onnx-to-onnx"; }
  StringRef getDescription() const final {
    return "Fold Add/Sub/Mul/Div through Q/DQ by updating scale/zero_point, "
           "then remove trivial Q->DQ chains when safe.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns
        .add<FoldBinaryThroughQDQ<ONNXDivOp>, FoldBinaryThroughQDQ<ONNXSubOp>,
            FoldBinaryThroughQDQ<ONNXMulOp>, FoldBinaryThroughQDQ<ONNXAddOp>>(
            &getContext());
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createFoldDQBinaryQPass() {
  return std::make_unique<FoldDQBinaryQPass>();
}
} // namespace onnx_mlir