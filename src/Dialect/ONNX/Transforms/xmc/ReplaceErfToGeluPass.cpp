// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass replaces the quantized Erf-based GELU subgraph with a single
// onnx.Gelu op.
//
// After the quant-types pass, the GELU subgraph appears as:
//   %x   = ...                    (quantized input)
//   %div = onnx.Div(%x, %sqrt2)
//   %erf = onnx.Erf(%div)
//   %add = onnx.Add(%erf, %one)
//   %mul = onnx.Mul(%x, %add)     x * (1 + erf(x / sqrt(2)))
//   %out = onnx.Mul(%mul, %half)  * 0.5
//
// This pattern is equivalent to gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
// The pass replaces it with:
//   onnx.Gelu(%x, approximate="none")
//
// The downstream ReplaceQDQEltwisePass then converts the quantized onnx.Gelu
// into XCOMPILERFusedEltwise(type="GELU").
// This mirrors the old xcompiler ReplaceErtToGeluPass logic.

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

#define DEBUG_TYPE "replace-erf-to-gelu"

using namespace mlir;

namespace {

static bool isQuantizedType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return isa<quant::QuantizedType>(tensorType.getElementType());
  if (auto tensorType = dyn_cast<UnrankedTensorType>(type))
    return isa<quant::QuantizedType>(tensorType.getElementType());
  return false;
}

// Match the Erf-based GELU subgraph and replace with onnx.Gelu.
//
// Pattern (after quant-types, all ops have !quant.uniform types):
//   Div(x, sqrt2) -> Erf -> Add(erf, 1) -> Mul(x, add) -> Mul(mul, 0.5)
//
// We anchor on ONNXErfOp and walk upstream/downstream to verify the pattern.
struct ReplaceErfGeluPattern : public OpRewritePattern<ONNXErfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXErfOp erfOp, PatternRewriter &rewriter) const override {
    if (!isQuantizedType(erfOp.getResult().getType()))
      return rewriter.notifyMatchFailure(erfOp, "erf result not quantized");

    // --- Upstream: Erf <- Div(x, sqrt2) ---
    auto divOp = erfOp.getInput().getDefiningOp<ONNXDivOp>();
    if (!divOp)
      return rewriter.notifyMatchFailure(erfOp, "erf input is not Div");

    Value geluInput = divOp.getA(); // The original x

    // --- Downstream: Erf -> Add(erf, const_1) ---
    Value erfResult = erfOp.getResult();
    if (!erfResult.hasOneUse())
      return rewriter.notifyMatchFailure(erfOp, "erf has multiple uses");

    auto addOp = dyn_cast<ONNXAddOp>(*erfResult.getUsers().begin());
    if (!addOp)
      return rewriter.notifyMatchFailure(erfOp, "erf user is not Add");

    // Verify erf result is one of Add's operands.
    if (addOp.getA() != erfResult && addOp.getB() != erfResult)
      return rewriter.notifyMatchFailure(erfOp, "erf not in Add operands");

    // --- Add -> Mul(x, add_result) ---
    Value addResult = addOp.getResult();
    if (!addResult.hasOneUse())
      return rewriter.notifyMatchFailure(erfOp, "add has multiple uses");

    auto mulOp = dyn_cast<ONNXMulOp>(*addResult.getUsers().begin());
    if (!mulOp)
      return rewriter.notifyMatchFailure(erfOp, "add user is not Mul");

    // Verify x and add_result are the two operands of Mul (either order).
    Value mulA = mulOp.getA();
    Value mulB = mulOp.getB();
    bool xInMul = (mulA == geluInput && mulB == addResult) ||
                  (mulB == geluInput && mulA == addResult);
    if (!xInMul)
      return rewriter.notifyMatchFailure(
          erfOp, "Mul operands don't match x and add");

    // --- Mul -> Mul1(mul_result, 0.5) ---
    Value mulResult = mulOp.getResult();
    if (!mulResult.hasOneUse())
      return rewriter.notifyMatchFailure(erfOp, "mul has multiple uses");

    auto mul1Op = dyn_cast<ONNXMulOp>(*mulResult.getUsers().begin());
    if (!mul1Op)
      return rewriter.notifyMatchFailure(erfOp, "mul user is not Mul(0.5)");

    LLVM_DEBUG(llvm::dbgs() << "replace-erf-to-gelu: Matched GELU pattern at "
                            << erfOp.getLoc() << "\n");

    // --- Create onnx.Gelu ---
    // Input: geluInput (x), output type: same as mul1's output
    Location loc = erfOp.getLoc();
    Type resultType = mul1Op.getResult().getType();

    auto geluOp = rewriter.create<ONNXGeluOp>(
        loc, resultType, geluInput, rewriter.getStringAttr("none"));

    rewriter.replaceOp(mul1Op, geluOp.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ReplaceErfToGeluPass
    : public PassWrapper<ReplaceErfToGeluPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "replace-erf-to-gelu"; }
  StringRef getDescription() const override {
    return "Replace quantized Erf-based GELU subgraph with onnx.Gelu";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReplaceErfGeluPattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createReplaceErfToGeluPass() {
  return std::make_unique<ReplaceErfToGeluPass>();
}

} // namespace onnx_mlir
