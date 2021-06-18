/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToKrnl.cpp - ONNX dialects to Krnl lowering -------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to a combination of
// Krnl IR and standard operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// EntryPoint Op lowering to Krnl Entry Point.
//===----------------------------------------------------------------------===//

class ONNXEntryPointLowering : public OpRewritePattern<ONNXEntryPointOp> {
public:
  using OpRewritePattern<ONNXEntryPointOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXEntryPointOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<KrnlEntryPointOp>(op,
        op->getAttrOfType<SymbolRefAttr>(
            ONNXEntryPointOp::getEntryPointFuncAttrName()),
        op->getAttrOfType<IntegerAttr>(
            ONNXEntryPointOp::getNumInputsAttrName()),
        op->getAttrOfType<IntegerAttr>(
            ONNXEntryPointOp::getNumOutputsAttrName()),
        op->getAttrOfType<StringAttr>(
            ONNXEntryPointOp::getSignatureAttrName()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Frontend to Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
namespace {
struct FrontendToKrnlLoweringPass
    : public PassWrapper<FrontendToKrnlLoweringPass, OperationPass<ModuleOp>> {
  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  FrontendToKrnlLoweringPass() = default;
  FrontendToKrnlLoweringPass(const FrontendToKrnlLoweringPass &pass) {}

  void runOnOperation() final;

public:
  // RNN ops are lowered to other ONNX ops such as ONNXMatMulOp, ONNXSplitOp,
  // ONNXTransposeOp, etc. These ONNX ops are then lowered into krnl ops in this
  // pass.
  //
  // To write LIT tests for RNN ops, we need not to check the final generated
  // krnl code that is lengthy but the intermediate generated code including
  // ONNX ops. We trust the lowering of the other ONNX ops.
  //
  // This flag is used in LIT tests to stop the lowering of the other ONNX ops.
  // Usage: onnx-mlir-opt --convert-onnx-to-krnl='check-rnn-ops-lowering'
  Option<bool> checkRNNOps{*this, "check-rnn-ops-lowering",
      llvm::cl::desc("Only used for writing LIT tests for RNN ops."),
      llvm::cl::init(false)};
};
} // end anonymous namespace.

void FrontendToKrnlLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<KrnlOpsDialect, AffineDialect, StandardOpsDialect,
      linalg::LinalgDialect, math::MathDialect, memref::MemRefDialect,
      shape::ShapeDialect, scf::SCFDialect>();

  // Use krnl.load/store instead of std.load/store and affine.load/store.
  // krnl.load/store will be lowered to std.load/store and affine.load/store by
  // `convert-krnl-to-affine` pass.
  target.addIllegalOp<mlir::memref::LoadOp>();
  target.addIllegalOp<mlir::AffineLoadOp>();
  target.addIllegalOp<mlir::memref::StoreOp>();
  target.addIllegalOp<mlir::AffineStoreOp>();

  // std.tanh will be expanded.
  target.addIllegalOp<mlir::math::TanhOp>();

  // TODO: enable this once more ops are supported.
  // We also define the ONNX dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted.
  // target.addIllegalDialect<mlir::ONNXOpsDialect>();

  // TODO: add any other ops which are considered legal.
  // Some operations can be marked as being still legal.
  // Example: target.addLegalOp<mlir::OpName>();

  if (checkRNNOps) {
    // Only used for writing LIT tests for RNN ops. We do not go further
    // lowering the following ops. See the comment in the declaration of
    // 'checkRNNOps' for more details.
    target.addLegalOp<ONNXTransposeOp>();
    target.addLegalOp<ONNXSqueezeOp>();
    target.addLegalOp<ONNXSplitOp>();
    target.addLegalOp<ONNXMatMulOp>();
    target.addLegalOp<ONNXSigmoidOp>();
    target.addLegalOp<ONNXTanhOp>();
  }

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

  // Convert TensorType to MemRef
  TensorTypeConverter tensorToMemRefConverter;
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    // FuncOp is legal only if types have been converted to Std types.
    return tensorToMemRefConverter.isSignatureLegal(op.getType());
  });

  target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
    // CallOp is legal only if types have been converted to Std types.
    return tensorToMemRefConverter.isLegal(op);
  });

  // Type conversion for function signatures.
  // Call MLIR FuncOp signature conversion when result type is
  // a ranked tensor.
  populateFuncOpTypeConversionPattern(patterns, tensorToMemRefConverter);
  populateCallOpTypeConversionPattern(patterns, tensorToMemRefConverter);

  // Frontend operation lowering.
  // ControlFlow
  populateLoweringONNXLoopOpPattern(patterns, &getContext());
  populateLoweringONNXScanOpPattern(patterns, &getContext());
  // Math
  populateLoweringONNXClipOpPattern(patterns, &getContext());
  populateLoweringONNXElementwiseOpPattern(patterns, &getContext());
  populateLoweringONNXGemmOpPattern(patterns, &getContext());
  populateLoweringONNXReductionOpPattern(patterns, &getContext());
  populateLoweringONNXSoftmaxOpPattern(patterns, &getContext());
  populateLoweringONNXMatMulOpPattern(patterns, &getContext());
  populateLoweringONNXLRNOpPattern(patterns, &getContext());
  // Tensor
  populateLoweringONNXArgMaxOpPattern(patterns, &getContext());
  populateLoweringONNXReshapeOpPattern(patterns, &getContext());
  populateLoweringONNXPadOpPattern(patterns, &getContext());
  populateLoweringONNXUnsqueezeOpPattern(patterns, &getContext());
  populateLoweringONNXTransposeOpPattern(patterns, &getContext());
  populateLoweringONNXGatherOpPattern(patterns, &getContext());
  populateLoweringONNXIdentityOpPattern(patterns, &getContext());
  populateLoweringONNXConstantOfShapeOpPattern(patterns, &getContext());
  populateLoweringONNXConstantOpPattern(patterns, &getContext());
  populateLoweringONNXConcatOpPattern(patterns, &getContext());
  populateLoweringONNXShapeOpPattern(patterns, &getContext());
  populateLoweringONNXSliceOpPattern(patterns, &getContext());
  populateLoweringONNXSqueezeOpPattern(patterns, &getContext());
  populateLoweringONNXSplitOpPattern(patterns, &getContext());
  populateLoweringONNXSizeOpPattern(patterns, &getContext());
  populateLoweringONNXTileOpPattern(patterns, &getContext());
  populateLoweringONNXFlattenOpPattern(patterns, &getContext());
  populateLoweringONNXResizeOpPattern(patterns, &getContext());
  // Neural network
  populateLoweringONNXConvOpPattern(patterns, &getContext());
  populateLoweringONNXNormalizationOpPattern(patterns, &getContext());
  populateLoweringONNXPoolingOpPattern(patterns, &getContext());
  // Recurrent neural network
  populateLoweringONNXGRUOpPattern(patterns, &getContext());
  populateLoweringONNXLSTMOpPattern(patterns, &getContext());
  populateLoweringONNXRNNOpPattern(patterns, &getContext());
  // Entry point
  patterns.insert<ONNXEntryPointLowering>(&getContext());

  // Expand std.tanh
  populateExpandTanhPattern(patterns);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createLowerToKrnlPass() {
  return std::make_unique<FrontendToKrnlLoweringPass>();
}
