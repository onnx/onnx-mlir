//===- ReplaceQDQClipCastPass.cpp - Fuse Clip+Cast to FusedEltwise CLAMP -===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Mirrors xcompiler ReplaceQDQClampCastPass for the post-quant-types ONNX IR:
//
//   onnx.Clip(quantized_input, min, max) -> f32
//        -> onnx.Cast(f32 -> uint)
//
// is fused into:
//
//   onnx.XCOMPILERFusedEltwise(type="CLAMP")
//
// with min/max taken from Clip operands and output quantization parameters
// y_scale=1.0, y_zero_point=0 (matching legacy xcompiler behavior).
//
//===----------------------------------------------------------------------===//

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

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "replace-qdq-clip-cast"

using namespace mlir;

namespace {

static bool isQuantizedType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return isa<quant::QuantizedType>(tensorType.getElementType());
  if (auto tensorType = dyn_cast<UnrankedTensorType>(type))
    return isa<quant::QuantizedType>(tensorType.getElementType());
  return false;
}

static bool isFloat32Type(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return tensorType.getElementType().isF32();
  if (auto tensorType = dyn_cast<UnrankedTensorType>(type))
    return tensorType.getElementType().isF32();
  return type.isF32();
}

static IntegerAttr getI32Attr(PatternRewriter &rewriter, int32_t value) {
  return rewriter.getI32IntegerAttr(value);
}

// Scalar onnx.Constant (splat or one element) as int64, for Clip min/max.
static std::optional<int64_t> getConstScalarI64(Value v) {
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
      return static_cast<int64_t>(std::llround(apf.convertToDouble()));
    }
    if (auto it = dyn_cast<IntegerType>(et)) {
      APInt api = elementsAttr.getSplatValue<APInt>();
      return static_cast<int64_t>(
          it.isUnsigned() ? api.getZExtValue() : api.getSExtValue());
    }
    return std::nullopt;
  }

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

// Build fused CLAMP output quant type. Raw uint Cast outputs are wrapped as
// !quant.uniform<uiN:f32, 1.0:0> so XFE→XIR can populate y_scale/y_zero_point
// (matches xcompiler ReplaceQDQClampCastPass).
static Type buildFusedClampCastResultType(
    MLIRContext *ctx, RankedTensorType castOutTy) {
  Type elemTy = castOutTy.getElementType();
  if (auto uq = dyn_cast<quant::UniformQuantizedType>(elemTy))
    return castOutTy;

  auto unsignedTy = dyn_cast<IntegerType>(elemTy);
  if (!unsignedTy || !unsignedTy.isUnsigned())
    return {};

  unsigned width = unsignedTy.getWidth();
  bool isSigned =
      unsignedTy.isSignedInteger() || unsignedTy.isSignlessInteger();
  auto quantElemTy = quant::UniformQuantizedType::get(isSigned, unsignedTy,
      Float32Type::get(ctx), 1.0, 0,
      quant::UniformQuantizedType::getDefaultMinimumForInteger(isSigned, width),
      quant::UniformQuantizedType::getDefaultMaximumForInteger(
          isSigned, width));
  return RankedTensorType::get(castOutTy.getShape(), quantElemTy);
}

// Fusion changes cast output from raw uint to quant.uniform; update the
// enclosing function signature before RAUW so mid-rewrite verification passes.
static void updateFuncReturnTypeForFusedCast(
    PatternRewriter &rewriter, Value castResult, Type fusedResultTy) {
  for (Operation *user : castResult.getUsers()) {
    auto returnOp = dyn_cast<func::ReturnOp>(user);
    if (!returnOp)
      continue;

    func::FuncOp funcOp = returnOp->getParentOfType<func::FuncOp>();
    FunctionType oldFuncTy = funcOp.getFunctionType();
    SmallVector<Type> newResultTypes(oldFuncTy.getResults());
    for (auto [idx, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (operand == castResult)
        newResultTypes[idx] = fusedResultTy;
    }
    rewriter.modifyOpInPlace(funcOp, [&]() {
      funcOp.setType(FunctionType::get(
          rewriter.getContext(), oldFuncTy.getInputs(), newResultTypes));
    });
  }
}

// onnx.Clip(quantized, min, max) -> f32 -> onnx.Cast(->uint) =>
// onnx.XCOMPILERFusedEltwise(type="CLAMP")
struct FuseQuantizedClipCastPattern : public OpRewritePattern<ONNXCastOp> {
  using OpRewritePattern<ONNXCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXCastOp castOp, PatternRewriter &rewriter) const override {
    auto clipOp = castOp.getInput().getDefiningOp<ONNXClipOp>();
    if (!clipOp)
      return rewriter.notifyMatchFailure(castOp, "input is not onnx.Clip");

    if (!clipOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(castOp, "clip has multiple users");

    if (!isQuantizedType(clipOp.getInput().getType()))
      return rewriter.notifyMatchFailure(castOp, "clip input is not quantized");

    if (!isFloat32Type(clipOp.getResult().getType()))
      return rewriter.notifyMatchFailure(
          castOp, "clip output is not float32 (expected DQ boundary)");

    if (!isFloat32Type(castOp.getInput().getType()))
      return rewriter.notifyMatchFailure(castOp, "cast input is not float32");

    auto castOutTy = dyn_cast<RankedTensorType>(castOp.getResult().getType());
    if (!castOutTy)
      return rewriter.notifyMatchFailure(castOp, "cast output is not ranked");

    Type fusedResultTy =
        buildFusedClampCastResultType(rewriter.getContext(), castOutTy);
    if (!fusedResultTy)
      return rewriter.notifyMatchFailure(
          castOp, "cast output is not uint or quantized");

    IntegerAttr clipMinAttr, clipMaxAttr;
    if (auto mn = getConstScalarI64(clipOp.getMin()))
      clipMinAttr = getI32Attr(rewriter, static_cast<int32_t>(*mn));
    else if (clipOp.getMin() && !isa<NoneType>(clipOp.getMin().getType()))
      return rewriter.notifyMatchFailure(castOp, "clip min not constant/none");

    if (auto mx = getConstScalarI64(clipOp.getMax()))
      clipMaxAttr = getI32Attr(rewriter, static_cast<int32_t>(*mx));
    else if (clipOp.getMax() && !isa<NoneType>(clipOp.getMax().getType()))
      return rewriter.notifyMatchFailure(castOp, "clip max not constant/none");

    if (!clipMinAttr && !clipMaxAttr)
      return rewriter.notifyMatchFailure(
          castOp, "no constant clip bounds to materialize");

    Value noneB = rewriter.create<ONNXNoneOp>(castOp.getLoc()).getResult();
    auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(castOp.getLoc(),
        fusedResultTy, clipOp.getInput(), noneB,
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

    updateFuncReturnTypeForFusedCast(
        rewriter, castOp.getResult(), fusedResultTy);
    rewriter.replaceOp(castOp, fusedOp.getResult());
    if (clipOp->use_empty())
      rewriter.eraseOp(clipOp);
    return success();
  }
};

struct ReplaceQDQClipCastPass
    : public PassWrapper<ReplaceQDQClipCastPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceQDQClipCastPass)

  StringRef getArgument() const override { return "replace-qdq-clip-cast"; }
  StringRef getDescription() const override {
    return "Fuse onnx.Clip(quantized->f32)+onnx.Cast(f32->uint) into "
           "onnx.XCOMPILERFusedEltwise CLAMP";
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<FuseQuantizedClipCastPattern>(context);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = 10;

    onnx_mlir::ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(applyPatternsAndFoldGreedily(
            function, std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createReplaceQDQClipCastPass() {
  return std::make_unique<ReplaceQDQClipCastPass>();
}
} // namespace onnx_mlir
