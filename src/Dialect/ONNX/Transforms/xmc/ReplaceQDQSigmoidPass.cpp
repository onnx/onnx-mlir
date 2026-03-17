// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass replaces quantized Sigmoid (and optionally HardSigmoid) with
// XCOMPILERFusedEltwise ops (QLINEARSIGMOID), matching the behavior of the
// XCompiler ReplaceQDQSigmoidPass.
//
// Patterns (registrations currently commented out):
// 1. Sigmoid with quantized in/out -> moved to replace-qdq-eltwise pass.
// 2. Sigmoid -> Mul(const) with quantized types -> same with mul_y (commented).
// 3. HardSigmoid is handled by replace-hsigmoid-and-hswish pass.
//
// Pass option: enable_lut_sigmoid sets the enable_lut_sigmoid attribute on
// created QLINEARSIGMOID ops (XCompiler parity).
//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "replace-qdq-sigmoid"

using namespace mlir;

namespace {

/// Helper to check that a type is quantized (has UniformQuantizedType element).
static LogicalResult requireQuantizedType(Type type) {
  auto tensorType = dyn_cast<TensorType>(type);
  if (!tensorType)
    return failure();
  return success(
      isa<mlir::quant::UniformQuantizedType>(tensorType.getElementType()));
}

/// Extract a scalar float from a constant value. Returns nullopt if not a
/// constant or not a single float.
static std::optional<float> getConstScalarF32(Value v) {
  if (!v || isa<NoneType>(v.getType()))
    return std::nullopt;
  auto cst = v.getDefiningOp<ONNXConstantOp>();
  if (!cst)
    return std::nullopt;
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(cst.getValueAttr());
  if (!elementsAttr)
    return std::nullopt;
  if (elementsAttr.isSplat()) {
    Type et = elementsAttr.getElementType();
    if (isa<FloatType>(et)) {
      APFloat apf = elementsAttr.getSplatValue<APFloat>();
      return static_cast<float>(apf.convertToDouble());
    }
    return std::nullopt;
  }
  auto shapedTy = dyn_cast<ShapedType>(elementsAttr.getType());
  if (!shapedTy || !shapedTy.hasStaticShape() || shapedTy.getNumElements() != 1)
    return std::nullopt;
  Attribute firstAttr = *elementsAttr.getValues<Attribute>().begin();
  if (auto f = dyn_cast<FloatAttr>(firstAttr))
    return static_cast<float>(f.getValueAsDouble());
  return std::nullopt;
}

/// Build XCOMPILERFusedEltwise QLINEARSIGMOID. Optional mul_y and
/// enable_lut_sigmoid (use FloatAttr() for mul_y when not set).
static XCOMPILERFusedEltwiseOp createQLinearSigmoidOp(PatternRewriter &rewriter,
    Location loc, Type outputType, Value inputA, Value inputB,
    std::optional<float> mulY, bool enableLutSigmoid) {
  FloatAttr mulYAttr =
      mulY ? rewriter.getFloatAttr(rewriter.getF32Type(), *mulY) : FloatAttr();
  BoolAttr enableLutAttr = rewriter.getBoolAttr(enableLutSigmoid);
  return rewriter.create<XCOMPILERFusedEltwiseOp>(loc, outputType, inputA,
      inputB,
      /*clip_max=*/IntegerAttr(),
      /*clip_min=*/IntegerAttr(), enableLutAttr,
      /*leakyrelu_alpha=*/FloatAttr(), mulYAttr,
      /*nonlinear=*/rewriter.getStringAttr("NONE"),
      /*nonlinear_in_scales=*/FloatAttr(),
      /*nonlinear_in_zeropoints=*/IntegerAttr(),
      /*prelu_in=*/IntegerAttr(),
      /*prelu_shift=*/IntegerAttr(),
      /*type=*/rewriter.getStringAttr("QLINEARSIGMOID"));
}

/// Replace Sigmoid -> Mul(const) (quantized) with XCOMPILERFusedEltwise
/// QLINEARSIGMOID with mul_y set from the constant. (Registration commented
/// out.)
struct ReplaceQuantizedSigmoidMulPattern : public OpRewritePattern<ONNXMulOp> {
  ReplaceQuantizedSigmoidMulPattern(MLIRContext *ctx, bool enableLutSigmoid)
      : OpRewritePattern<ONNXMulOp>(ctx), enableLutSigmoid(enableLutSigmoid) {}

  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const override {
    Value a = mulOp.getA();
    Value b = mulOp.getB();
    ONNXSigmoidOp sigmoidOp = a.getDefiningOp<ONNXSigmoidOp>();
    Value constVal = b;
    if (!sigmoidOp) {
      sigmoidOp = b.getDefiningOp<ONNXSigmoidOp>();
      constVal = a;
    }
    if (!sigmoidOp)
      return failure();

    std::optional<float> mulY = getConstScalarF32(constVal);
    if (!mulY)
      return rewriter.notifyMatchFailure(
          mulOp, "Mul other operand is not a constant scalar float");

    Value sigmoidInput = sigmoidOp.getX();
    Type outputType = mulOp.getResult().getType();
    if (failed(requireQuantizedType(sigmoidInput.getType())))
      return rewriter.notifyMatchFailure(
          mulOp, "Sigmoid input is not quantized");
    if (failed(requireQuantizedType(outputType)))
      return rewriter.notifyMatchFailure(mulOp, "Mul output is not quantized");

    Location loc = mulOp.getLoc();
    auto noneOp =
        rewriter.create<ONNXNoneOp>(loc, rewriter.getNoneType(), true);

    auto fusedEltwiseOp = createQLinearSigmoidOp(rewriter, loc, outputType,
        sigmoidInput, noneOp.getResult(), mulY, enableLutSigmoid);

    rewriter.replaceOp(mulOp, fusedEltwiseOp.getResult());
    return success();
  }
  bool enableLutSigmoid;
};

} // namespace

namespace onnx_mlir {

struct ReplaceQDQSigmoidPass
    : public PassWrapper<ReplaceQDQSigmoidPass, OperationPass<func::FuncOp>> {
  ReplaceQDQSigmoidPass() = default;
  ReplaceQDQSigmoidPass(const ReplaceQDQSigmoidPass &pass)
      : PassWrapper(pass) {}

  Option<bool> enableLutSigmoid{*this, "enable-lut-sigmoid",
      llvm::cl::desc("Set enable_lut_sigmoid on QLINEARSIGMOID ops"),
      llvm::cl::init(false)};

  StringRef getArgument() const override { return "replace-qdq-sigmoid"; }
  StringRef getDescription() const override {
    return "Replace quantized Sigmoid (and Sigmoid+Mul) with "
           "XCOMPILERFusedEltwise QLINEARSIGMOID";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    // Simple sigmoid pattern moved to replace-qdq-eltwise pass.
    // to do: enable these advancedpatterns as when required
    // patterns.add<ReplaceQuantizedSigmoidPattern>(ctx, enableLutSigmoid);
    // patterns.add<ReplaceQuantizedSigmoidMulPattern>(ctx, enableLutSigmoid);
    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createReplaceQDQSigmoidPass() {
  return std::make_unique<ReplaceQDQSigmoidPass>();
}

} // namespace onnx_mlir
