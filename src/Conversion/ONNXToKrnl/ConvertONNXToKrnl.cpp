/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToKrnl.cpp - ONNX dialects to Krnl lowering -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to a combination of
// Krnl IR and standard operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;
using namespace onnx_mlir;

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

namespace onnx_mlir {
void populateONNXToKrnlConversionPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableTiling) {
  // Type conversion for function signatures.
  // Call MLIR FuncOp signature conversion when result type is
  // a ranked tensor.
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(
      patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  // Frontend operation lowering.
  // ControlFlow
  onnx_mlir::populateLoweringONNXLoopOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXScanOpPattern(patterns, typeConverter, ctx);
  // Math
  onnx_mlir::populateLoweringONNXClipOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXCumSumOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXElementwiseOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXGemmOpPattern(
      patterns, typeConverter, ctx, enableTiling);
  onnx_mlir::populateLoweringONNXHardmaxOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXReductionOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSoftmaxOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXTopKOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXMatMulOpPattern(
      patterns, typeConverter, ctx, enableTiling);
  onnx_mlir::populateLoweringONNXRandomNormalOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXRandomNormalLikeOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXLRNOpPattern(patterns, typeConverter, ctx);
  // ML
  onnx_mlir::populateLoweringONNXCategoryMapperOpPattern(
      patterns, typeConverter, ctx);
  // ObjectDetection
  onnx_mlir::populateLoweringONNXNonMaxSuppressionOpPattern(
      patterns, typeConverter, ctx);
  // Tensor
  onnx_mlir::populateLoweringONNXArgMaxOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXReshapeOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXPadOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXUnsqueezeOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXUnsqueezeV11OpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXTransposeOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXGatherOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXIdentityOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXConstantOfShapeOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXConstantOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXConcatOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXDepthToSpaceOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSpaceToDepthOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXShapeOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSliceOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSqueezeOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSqueezeV11OpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSplitOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSplitV11OpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSizeOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXTileOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXFlattenOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXRangeOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXResizeOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXNonZeroOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXReverseSequenceOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXExpandOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXOneHotOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXCompressOpPattern(
      patterns, typeConverter, ctx);
  // Neural network
  onnx_mlir::populateLoweringONNXConvOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXNormalizationOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXPoolingOpPattern(patterns, typeConverter, ctx);
  // Recurrent neural network
  onnx_mlir::populateLoweringONNXGRUOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXLSTMOpPattern(patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXRNNOpPattern(patterns, typeConverter, ctx);
  // Sequence
  onnx_mlir::populateLoweringONNXSequenceAtOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSequenceEmptyOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSequenceEraseOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSequenceInsertOpPattern(
      patterns, typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSequenceLengthOpPattern(
      patterns, typeConverter, ctx);
  // Entry point
  patterns.insert<ONNXEntryPointLowering>(ctx);
}
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Frontend to Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
namespace {
struct FrontendToKrnlLoweringPass
    : public PassWrapper<FrontendToKrnlLoweringPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-onnx-to-krnl"; }

  StringRef getDescription() const override {
    return "Lower frontend ops to Krnl dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  FrontendToKrnlLoweringPass() = default;
  FrontendToKrnlLoweringPass(const FrontendToKrnlLoweringPass &pass)
      : PassWrapper<FrontendToKrnlLoweringPass, OperationPass<ModuleOp>>() {}
  FrontendToKrnlLoweringPass(bool emitDealloc, bool enableTiling) {
    // Below, need explicit assignment to enable implicit conversion of bool to
    // Option<bool>.
    this->emitDealloc = emitDealloc;
    this->enableTiling = enableTiling;
  }
  FrontendToKrnlLoweringPass(int optLevel)
      : FrontendToKrnlLoweringPass(
            /*emitDealloc=*/false, /*enableTiling=*/optLevel >= 3) {}

  void runOnOperation() final;

public:
  // Some ops (RNN ops for example) are lowered to other ONNX ops such as
  // ONNXMatMulOp, ONNXSplitOp, ONNXTransposeOp, etc. These ONNX ops are then
  // lowered into krnl ops in this pass.
  //
  // To write LIT tests for operations that are lowered to other ONNX
  // operations, we do not need to check the final generated krnl code (which is
  // lengthy). It is more convenient to check the intermediate generated code
  // including ONNX ops. We trust the lowering of the other ONNX ops.
  //
  // This flag is used in LIT tests to stop the lowering of the other ONNX ops.
  // Usage: onnx-mlir-opt --convert-onnx-to-krnl='emit-intermediate-ir'
  Option<bool> emitIntermediateIR{*this, "emit-intermediate-ir",
      llvm::cl::desc(
          "Emit intermediate IR rather than lowering to the krnl dialect."),
      llvm::cl::init(false)};
  Option<bool> emitDealloc{*this, "emit-dealloc",
      llvm::cl::desc("Emit dealloc for allocated memrefs or not."),
      llvm::cl::init(false)};
  Option<bool> enableTiling{*this, "enable-tiling",
      llvm::cl::desc("Enable loop tiling and unrolling optimizations"),
      llvm::cl::init(false)};
};
} // end anonymous namespace.

void FrontendToKrnlLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Set up whether emitting dealloc for allocated memrefs or not.
  ONNXToKrnl_gEmitDealloc = emitDealloc;

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target
      .addLegalDialect<KrnlOpsDialect, AffineDialect, arith::ArithmeticDialect,
          StandardOpsDialect, linalg::LinalgDialect, math::MathDialect,
          memref::MemRefDialect, shape::ShapeDialect, scf::SCFDialect>();
  // Needed to support unsigned int computations. To be removed if we use a
  // scheme that does not rely on the UnrealizedConversionCastOp.
  target.addLegalOp<::mlir::UnrealizedConversionCastOp>();
  // Make ONNXNoneOp legal so that other ONNX ops can use it during the
  // lowering. ONNXNoneOp will be dangling and removed by calling
  // canonicalization after the lowering.
  target.addLegalOp<::mlir::ONNXNoneOp>();

  // Use krnl.load/store instead of std.load/store and affine.load/store.
  // krnl.load/store will be lowered to std.load/store and affine.load/store by
  // `convert-krnl-to-affine` pass.
  target.addIllegalOp<mlir::memref::LoadOp>();
  target.addIllegalOp<mlir::AffineLoadOp>();
  target.addIllegalOp<mlir::memref::StoreOp>();
  target.addIllegalOp<mlir::AffineStoreOp>();

  // If `emitDealloc` is turned off, make sure we don't have buffer deallocation
  // at this level. Will use MLIR buffer-deallocation for this purpose instead.
  if (!emitDealloc)
    target.addIllegalOp<mlir::memref::DeallocOp>();

  // TODO: enable this once more ops are supported.
  // We also define the ONNX dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted.
  // target.addIllegalDialect<mlir::ONNXOpsDialect>();

  // TODO: add any other ops which are considered legal.
  // Some operations can be marked as being still legal.
  // Example: target.addLegalOp<mlir::OpName>();

  if (emitIntermediateIR) {
    // Only used for writing LIT tests for ONNX operations that are lowered to
    // other ONNX operations. The following operations are prevented from being
    // lowered further. See the comment in the declaration of
    // 'emitIntermediateIR' for more details.
    target.addLegalOp<ONNXMatMulOp>();
    target.addLegalOp<ONNXReshapeOp>();
    target.addLegalOp<ONNXSplitV11Op>();
    target.addLegalOp<ONNXSqueezeV11Op>();
    target.addLegalOp<ONNXTransposeOp>();
  }

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

  // Convert types to legal types for the Krnl dialect.
  KrnlTypeConverter krnlTypeConverter;
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    // FuncOp is legal only if types have been converted to Std types.
    return krnlTypeConverter.isSignatureLegal(op.getType());
  });

  target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
    // CallOp is legal only if types have been converted to Std types.
    return krnlTypeConverter.isLegal(op);
  });

  // Operations that are legal only if types are not tensors.
  target.addDynamicallyLegalOp<mlir::ReturnOp>([&](Operation *op) {
    return llvm::none_of(op->getOperandTypes(),
        [](Type type) { return type.isa<TensorType>(); });
  });

  // Define patterns.
  populateONNXToKrnlConversionPattern(
      patterns, krnlTypeConverter, &getContext(), enableTiling);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> onnx_mlir::createLowerToKrnlPass() {
  return std::make_unique<FrontendToKrnlLoweringPass>();
}

std::unique_ptr<Pass> onnx_mlir::createLowerToKrnlPass(int optLevel) {
  return std::make_unique<FrontendToKrnlLoweringPass>(optLevel);
}

std::unique_ptr<Pass> onnx_mlir::createLowerToKrnlPass(
    bool emitDealloc, bool enableTiling) {
  return std::make_unique<FrontendToKrnlLoweringPass>(
      emitDealloc, enableTiling);
}
