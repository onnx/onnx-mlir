// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass replaces quantized HardSigmoid operations with
// XCOMPILERFusedEltwise ops that work directly with quantized tensor types.
// TODO: Replacing HSwish
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
#define DEBUG_TYPE "replace-hsigmoid-and-hswish"

using namespace mlir;

namespace {

/// Helper function to extract quantization parameters from a tensor type
/// Returns failure if the type is not quantized
static LogicalResult extractQuantParamsFromType(
    Type type, double &scale, int64_t &zeroPoint) {
  auto tensorType = dyn_cast<TensorType>(type);
  if (!tensorType)
    return failure();

  auto quantType =
      dyn_cast<mlir::quant::UniformQuantizedType>(tensorType.getElementType());
  if (!quantType)
    return failure();

  scale = quantType.getScale();
  zeroPoint = quantType.getZeroPoint();
  return success();
}

/// Pattern to match HardSigmoid with quantized input/output types
/// and replace with XCOMPILERFusedEltwise with type = "HSIGMOID"
///
/// The transformation is:
///   Input (quant type) -> HardSigmoid -> Output (quant type)
/// becomes:
///   Input (quant type) -> XCOMPILERFusedEltwise -> Output (quant type)
///
/// XCOMPILERFusedEltwise directly accepts and produces quantized tensor types,
/// with quantization parameters stored in qscales/qzeropoints attributes.
struct ReplaceQuantizedHardSigmoidPattern
    : public OpRewritePattern<ONNXHardSigmoidOp> {
  using OpRewritePattern<ONNXHardSigmoidOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXHardSigmoidOp hardSigmoidOp,
      PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "replace-hsigmoid-and-hswish: Trying to match "
                            << hardSigmoidOp << "\n");
    Value input = hardSigmoidOp.getX();
    Value output = hardSigmoidOp.getY();

    // Extract quantization parameters from input type
    double inputScale;
    int64_t inputZeroPoint;
    if (failed(extractQuantParamsFromType(
            input.getType(), inputScale, inputZeroPoint))) {
      return rewriter.notifyMatchFailure(
          hardSigmoidOp, "Input does not have quantized type");
    }

    // Extract quantization parameters from output type
    double outputScale;
    int64_t outputZeroPoint;
    if (failed(extractQuantParamsFromType(
            output.getType(), outputScale, outputZeroPoint))) {
      return rewriter.notifyMatchFailure(
          hardSigmoidOp, "Output does not have quantized type");
    }

    Location loc = hardSigmoidOp.getLoc();

    // Create a None value for the second operand (B) since HardSigmoid is unary
    auto noneOp =
        rewriter.create<ONNXNoneOp>(loc, rewriter.getNoneType(), true);

    // Create XCOMPILERFusedEltwise op with type = "HSIGMOID"
    // Directly uses quantized tensor types - no scast needed
    auto fusedEltwiseOp = rewriter.create<XCOMPILERFusedEltwiseOp>(loc,
        output.getType(),   // Output type (quant tensor)
        input,              // A - quantized tensor input
        noneOp.getResult(), // B - none for unary op
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
        /*type=*/rewriter.getStringAttr("HSIGMOID"));

    if (FloatAttr alphaAttr = hardSigmoidOp.getAlphaAttr())
      fusedEltwiseOp->setAttr("alpha", alphaAttr);
    if (FloatAttr betaAttr = hardSigmoidOp.getBetaAttr())
      fusedEltwiseOp->setAttr("beta", betaAttr);

    // Replace HardSigmoid directly with XCOMPILERFusedEltwise output
    rewriter.replaceOp(hardSigmoidOp, fusedEltwiseOp.getResult());

    return success();
  }
};

/// Extract a scalar float from a constant value. Returns nullopt if the value
/// is not produced by an ONNXConstantOp holding a single float element.
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

/// Pattern to fuse HardSigmoid + Mul into XCOMPILERFusedEltwise (HSIGMOID).
///
/// Handles Mul(const, HardSigmoid(x)) by extracting const as mul_y.
///
/// Creates: XCOMPILERFusedEltwise with type="HSIGMOID", alpha/beta from
/// HardSigmoid, and optional mul_y from constant operand.
struct FuseHardSigmoidMulToFusedEltwisePattern
    : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const override {
    Value a = mulOp.getA();
    Value b = mulOp.getB();

    // Find HardSigmoid as one operand of the Mul.
    ONNXHardSigmoidOp hardSigmoidOp = a.getDefiningOp<ONNXHardSigmoidOp>();
    Value otherOperand = b;
    if (!hardSigmoidOp) {
      hardSigmoidOp = b.getDefiningOp<ONNXHardSigmoidOp>();
      otherOperand = a;
    }
    if (!hardSigmoidOp)
      return failure();

    Value hsigmoidInput = hardSigmoidOp.getX();
    Type mulOutputType = mulOp.getResult().getType();

    double inputScale, outputScale;
    int64_t inputZeroPoint, outputZeroPoint;
    if (failed(extractQuantParamsFromType(
            hsigmoidInput.getType(), inputScale, inputZeroPoint)))
      return rewriter.notifyMatchFailure(
          mulOp, "HardSigmoid input is not quantized");
    if (failed(extractQuantParamsFromType(
            mulOutputType, outputScale, outputZeroPoint)))
      return rewriter.notifyMatchFailure(mulOp, "Mul output is not quantized");

    FloatAttr alphaAttr = hardSigmoidOp.getAlphaAttr();
    FloatAttr betaAttr = hardSigmoidOp.getBetaAttr();

    std::optional<float> mulY = getConstScalarF32(otherOperand);
    if (!mulY)
      return rewriter.notifyMatchFailure(
          mulOp, "non-HardSigmoid Mul operand is not a constant scalar float");
    FloatAttr mulYAttr = rewriter.getFloatAttr(rewriter.getF32Type(), *mulY);

    Location loc = mulOp.getLoc();
    auto noneOp =
        rewriter.create<ONNXNoneOp>(loc, rewriter.getNoneType(), true);

    auto fusedOp = rewriter.create<XCOMPILERFusedEltwiseOp>(loc, mulOutputType,
        hsigmoidInput, noneOp.getResult(),
        /*approximate=*/StringAttr(),
        /*enable_lut_sigmoid=*/rewriter.getBoolAttr(false),
        /*leakyrelu_alpha=*/FloatAttr(),
        /*max=*/IntegerAttr(),
        /*min=*/IntegerAttr(),
        /*mul_y=*/mulYAttr,
        /*nonlinear=*/rewriter.getStringAttr("NONE"),
        /*nonlinear_in_scales=*/FloatAttr(),
        /*nonlinear_in_zeropoints=*/IntegerAttr(),
        /*prelu_in=*/IntegerAttr(),
        /*prelu_shift=*/IntegerAttr(),
        /*type=*/rewriter.getStringAttr("HSIGMOID"));

    if (alphaAttr)
      fusedOp->setAttr("alpha", alphaAttr);
    if (betaAttr)
      fusedOp->setAttr("beta", betaAttr);

    rewriter.replaceOp(mulOp, fusedOp.getResult());
    if (hardSigmoidOp->use_empty())
      rewriter.eraseOp(hardSigmoidOp);

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ReplaceHsigmoidAndHswishPass
    : public PassWrapper<ReplaceHsigmoidAndHswishPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "replace-hsigmoid-and-hswish";
  }
  StringRef getDescription() const override {
    return "Replace quantized HardSigmoid operations with "
           "XCOMPILERFusedEltwise ops";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReplaceQuantizedHardSigmoidPattern>(context);
    patterns.add<FuseHardSigmoidMulToFusedEltwisePattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createReplaceHsigmoidAndHswishPass() {
  return std::make_unique<ReplaceHsigmoidAndHswishPass>();
}

} // namespace onnx_mlir
