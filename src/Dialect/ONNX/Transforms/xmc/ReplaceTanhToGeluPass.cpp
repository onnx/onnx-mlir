// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass replaces the quantized tanh-approximation GELU subgraph with a
// single onnx.Gelu op (approximate="tanh").
//
// After the quant-types pass, the tanh-GELU subgraph appears as:
//   %pow  = onnx.Pow(%x, %three)               // x^3
//   %mul1 = onnx.Mul(%c_0_044715, %pow)        // 0.044715 * x^3
//   %add1 = onnx.Add(%x, %mul1)                // x + 0.044715 * x^3
//   %mul2 = onnx.Mul(%c_0_79788, %add1)        // sqrt(2/pi) * (...)
//   %tanh = onnx.Tanh(%mul2)
//   %add2 = onnx.Add(%c_one, %tanh)            // 1 + tanh(...)
//   %mul3 = onnx.Mul(%x, %add2)                // x * (1 + tanh(...))
//   %mul4 = onnx.Mul(%c_half, %mul3)           // 0.5 * x * (1 + tanh(...))
//
// This pattern is equivalent to
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
// The pass replaces it with:
//   onnx.Gelu(%x, approximate="tanh")
//
// The downstream ReplaceQDQEltwisePass then converts the quantized onnx.Gelu
// into XCOMPILERFusedEltwise(type="GELU").
// Mirrors ReplaceErfToGeluPass but for the tanh approximation.

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

#define DEBUG_TYPE "replace-tanh-to-gelu"

using namespace mlir;

namespace {

static bool isQuantizedType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return isa<quant::QuantizedType>(tensorType.getElementType());
  if (auto tensorType = dyn_cast<UnrankedTensorType>(type))
    return isa<quant::QuantizedType>(tensorType.getElementType());
  return false;
}

// Read the scalar floating-point value out of an onnx.Constant feeding a
// scalar/broadcastable operand. The constant may be:
//   - a plain f32 ConstantOp (unquantized; common for Pow exponents); or
//   - a ui8/i8 ConstantOp wrapped in a !quant.uniform<...> type, in which case
//     we de-quantize it as f = (raw - zeroPoint) * scale.
// Returns std::nullopt on anything else (non-splat, multi-element, etc.).
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

  // Unquantized float constant.
  Type elemType = dense.getType().getElementType();
  if (isa<FloatType>(elemType)) {
    return dense.getSplatValue<APFloat>().convertToDouble();
  }

  // Quantized integer constant: dequantize using the result's quant type.
  auto resultType = dyn_cast<RankedTensorType>(constOp.getResult().getType());
  if (!resultType)
    return std::nullopt;
  auto qType =
      dyn_cast<quant::UniformQuantizedType>(resultType.getElementType());
  if (!qType)
    return std::nullopt;
  if (!isa<IntegerType>(elemType))
    return std::nullopt;
  int64_t raw = qType.isSigned() ? dense.getSplatValue<APInt>().getSExtValue()
                                 : dense.getSplatValue<APInt>().getZExtValue();
  return (static_cast<double>(raw) - qType.getZeroPoint()) * qType.getScale();
}

static bool isScalarConstantNear(Value v, double target, double tol = 1e-2) {
  auto fp = readScalarConstant(v);
  if (!fp)
    return false;
  return std::abs(*fp - target) <= tol;
}

static bool isScalarConstant(Value v) {
  return readScalarConstant(v).has_value();
}

// Given a binary op with one constant and one non-constant operand, return
// the non-constant operand. Returns nullptr if neither (or both) operands are
// constants.
template <typename BinaryOp>
static Value pickNonConstOperand(BinaryOp op) {
  Value a = op.getA();
  Value b = op.getB();
  bool aConst = isScalarConstant(a);
  bool bConst = isScalarConstant(b);
  if (aConst == bConst)
    return nullptr;
  return aConst ? b : a;
}

// Match: Add(x, Mul(c, Pow(x, 3))) -- the inner `x + 0.044715 * x^3` of the
// tanh-GELU subgraph. On success, populate `x` and `powOp`.
static bool matchAddOfXAndMulPow(ONNXAddOp addOp, Value &x, ONNXPowOp &powOp) {
  Value a = addOp.getA();
  Value b = addOp.getB();
  // Try both orderings: (x, Mul) or (Mul, x).
  for (auto [xCand, mulCand] :
      std::array<std::pair<Value, Value>, 2>{{{a, b}, {b, a}}}) {
    auto mul = mulCand.getDefiningOp<ONNXMulOp>();
    if (!mul)
      continue;
    Value powCand = pickNonConstOperand(mul);
    if (!powCand)
      continue;
    auto pow = powCand.getDefiningOp<ONNXPowOp>();
    if (!pow)
      continue;
    // Verify Pow's exponent is approximately 3.0.
    if (!isScalarConstantNear(pow.getY(), 3.0))
      continue;
    // Verify Pow's base is the same x as the Add's other operand.
    if (pow.getX() != xCand)
      continue;
    x = xCand;
    powOp = pow;
    return true;
  }
  return false;
}

// Match the tanh-approximation GELU subgraph anchored on ONNXTanhOp and
// replace the outermost Mul (`0.5 * x * (...)`) with onnx.Gelu(x, "tanh").
struct ReplaceTanhGeluPattern : public OpRewritePattern<ONNXTanhOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXTanhOp tanhOp, PatternRewriter &rewriter) const override {
    if (!isQuantizedType(tanhOp.getResult().getType()))
      return rewriter.notifyMatchFailure(tanhOp, "tanh result not quantized");

    // --- Upstream: Tanh <- Mul2(c_sqrt2pi, Add1(x, Mul1(c_044715, Pow(x, 3))))
    // ---
    auto mul2Op = tanhOp.getInput().getDefiningOp<ONNXMulOp>();
    if (!mul2Op)
      return rewriter.notifyMatchFailure(tanhOp, "tanh input is not Mul");
    Value mul2NonConst = pickNonConstOperand(mul2Op);
    if (!mul2NonConst)
      return rewriter.notifyMatchFailure(
          tanhOp, "Mul before Tanh needs one constant operand");

    auto add1Op = mul2NonConst.getDefiningOp<ONNXAddOp>();
    if (!add1Op)
      return rewriter.notifyMatchFailure(
          tanhOp, "Mul before Tanh does not feed from Add");

    Value x;
    ONNXPowOp powOp;
    if (!matchAddOfXAndMulPow(add1Op, x, powOp))
      return rewriter.notifyMatchFailure(
          tanhOp, "Add is not (x + c * Pow(x, 3))");

    // --- Downstream: Tanh -> Add2(c_one, tanh) -> Mul3(x, add2) ->
    // Mul4(c_half, mul3) ---
    Value tanhResult = tanhOp.getResult();
    if (!tanhResult.hasOneUse())
      return rewriter.notifyMatchFailure(tanhOp, "tanh has multiple uses");

    auto add2Op = dyn_cast<ONNXAddOp>(*tanhResult.getUsers().begin());
    if (!add2Op)
      return rewriter.notifyMatchFailure(tanhOp, "tanh user is not Add");
    // The non-tanh operand of add2 must be the constant 1.0.
    Value add2Other;
    if (add2Op.getA() == tanhResult)
      add2Other = add2Op.getB();
    else if (add2Op.getB() == tanhResult)
      add2Other = add2Op.getA();
    else
      return rewriter.notifyMatchFailure(tanhOp, "tanh not in Add2 operands");
    if (!isScalarConstantNear(add2Other, 1.0))
      return rewriter.notifyMatchFailure(
          tanhOp, "Add after Tanh must add constant 1.0");

    Value add2Result = add2Op.getResult();
    if (!add2Result.hasOneUse())
      return rewriter.notifyMatchFailure(tanhOp, "add2 has multiple uses");

    // The outer Mul applied to (1 + tanh(...)). Two associativity variants:
    //   Variant A: outerMul = Mul(x, add2);  finalMul = Mul(0.5, outerMul)
    //   Variant B: outerMul = Mul(add2, Mul(x, 0.5));  finalMul = outerMul
    auto outerMul = dyn_cast<ONNXMulOp>(*add2Result.getUsers().begin());
    if (!outerMul)
      return rewriter.notifyMatchFailure(tanhOp, "add2 user is not Mul");
    Value outerOther =
        (outerMul.getA() == add2Result) ? outerMul.getB() : outerMul.getA();

    ONNXMulOp finalMul;
    if (outerOther == x) {
      // Variant A: trailing 0.5 multiply.
      if (!outerMul.getResult().hasOneUse())
        return rewriter.notifyMatchFailure(
            tanhOp, "x*(1+tanh) has multiple uses");
      auto trailing =
          dyn_cast<ONNXMulOp>(*outerMul.getResult().getUsers().begin());
      if (!trailing)
        return rewriter.notifyMatchFailure(
            tanhOp, "x*(1+tanh) user is not Mul");
      Value trailingOther = (trailing.getA() == outerMul.getResult())
                                ? trailing.getB()
                                : trailing.getA();
      if (!isScalarConstantNear(trailingOther, 0.5))
        return rewriter.notifyMatchFailure(
            tanhOp, "trailing Mul must multiply by 0.5");
      finalMul = trailing;
    } else {
      // Variant B: outerOther must be Mul(x, 0.5) in either order.
      auto preMul = outerOther.getDefiningOp<ONNXMulOp>();
      if (!preMul)
        return rewriter.notifyMatchFailure(
            tanhOp, "outer Mul's non-(1+tanh) operand is not x or Mul(x,0.5)");
      bool preMatches =
          (preMul.getA() == x && isScalarConstantNear(preMul.getB(), 0.5)) ||
          (preMul.getB() == x && isScalarConstantNear(preMul.getA(), 0.5));
      if (!preMatches)
        return rewriter.notifyMatchFailure(
            tanhOp, "expected pre-Mul to be Mul(x, 0.5) in some order");
      finalMul = outerMul;
    }

    LLVM_DEBUG(llvm::dbgs() << "replace-tanh-to-gelu: matched tanh-GELU at "
                            << tanhOp.getLoc() << "\n");

    // --- Build onnx.Gelu(x, approximate="tanh") ---
    Location loc = tanhOp.getLoc();
    Type resultType = finalMul.getResult().getType();
    auto geluOp = rewriter.create<ONNXGeluOp>(
        loc, resultType, x, rewriter.getStringAttr("tanh"));
    rewriter.replaceOp(finalMul, geluOp.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ReplaceTanhToGeluPass
    : public PassWrapper<ReplaceTanhToGeluPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "replace-tanh-to-gelu"; }
  StringRef getDescription() const override {
    return "Replace quantized tanh-approximation GELU subgraph with onnx.Gelu";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReplaceTanhGeluPattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createReplaceTanhToGeluPass() {
  return std::make_unique<ReplaceTanhToGeluPass>();
}

} // namespace onnx_mlir
