// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass replaces quantized HardSigmoid operations with
// XCOMPILERFusedEltwise ops that work directly with quantized tensor types.
// TODO: Replacing HSwish
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
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
/// and replace with XCOMPILERFusedEltwise with type = "QLINEARSIGMOID"
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

    // Create XCOMPILERFusedEltwise op with type = "QLINEARSIGMOID"
    // Directly uses quantized tensor types - no scast needed
    auto fusedEltwiseOp = rewriter.create<XCOMPILERFusedEltwiseOp>(loc,
        output.getType(),   // Output type (quant tensor)
        input,              // A - quantized tensor input
        noneOp.getResult(), // B - none for unary op
        /*clip_max=*/IntegerAttr(),
        /*clip_min=*/IntegerAttr(),
        /*leakyrelu_alpha=*/FloatAttr(),
        /*nonlinear=*/rewriter.getStringAttr("NONE"),
        /*nonlinear_in_scales=*/FloatAttr(),
        /*nonlinear_in_zeropoints=*/IntegerAttr(),
        /*prelu_in=*/IntegerAttr(),
        /*prelu_shift=*/IntegerAttr(),
        /*type=*/rewriter.getStringAttr("QLINEARSIGMOID"));

    // Replace HardSigmoid directly with XCOMPILERFusedEltwise output
    rewriter.replaceOp(hardSigmoidOp, fusedEltwiseOp.getResult());

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
    return "Replace quantized HardSigmoid and HSwish operations with "
           "XCOMPILERFusedEltwise ops";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReplaceQuantizedHardSigmoidPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createReplaceHsigmoidAndHswishPass() {
  return std::make_unique<ReplaceHsigmoidAndHswishPass>();
}

} // namespace onnx_mlir
