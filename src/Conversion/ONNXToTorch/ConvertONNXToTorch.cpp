/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====- ConvertONNXToTorch.cpp - ONNX dialects to Torch lowering -===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// ========================================================================
//
// This file implements the lowering of frontend operations to a combination
// of Torch IR and standard operations.
//
//===------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

void populateONNXToTorchConversionPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableTiling) {

  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  populateLoweringONNXToTorchConvOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchConstantPadNdOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchLeakyReluOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchMaxPoolSingleOutOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchConstOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchReduceMeanOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchGemmOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchSoftmaxOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchAddOpPattern (patterns, typeConverter, ctx);
  populateLoweringONNXToTorchFlattenOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchElementwiseOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchSqueezeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchConcatOpPattern (patterns, typeConverter, ctx);
  populateLoweringONNXToTorchBinaryOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXToTorchArgmaxOpPattern(patterns, typeConverter, ctx);
}

//===----------------------------------------------------------------------===//
// FinalizingBackendTypeConversionPass
//===----------------------------------------------------------------------===//

namespace {
// In a finalizing conversion, we know that all of the source types have been
// converted to the destination types, so the materialization becomes an
// identity.
template <typename OpTy>
class FinalizeMaterialization : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};
} // namespace

template <typename OpTy>
static void setupFinalization(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              TypeConverter &typeConverter) {
  target.addIllegalOp<OpTy>();
  patterns.add<FinalizeMaterialization<OpTy>>(typeConverter,
                                              patterns.getContext());
}

template <typename OpTy, typename OpTy2, typename... OpTys>
static void setupFinalization(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              TypeConverter &typeConverter) {
  setupFinalization<OpTy>(target, patterns, typeConverter);
  setupFinalization<OpTy2, OpTys...>(target, patterns, typeConverter);
}

//===-----------------------------------------------------------------===//
// Frontend to Torch Dialect lowering pass
//===-----------------------------------------------------------------===//

/// This is a partial lowering to Torch loops of the ONNX operations.
namespace onnx_mlir {
struct FrontendToTorchLoweringPass
    : public PassWrapper<FrontendToTorchLoweringPass,
          OperationPass<::mlir::ModuleOp>> {

  StringRef getArgument() const override { return "convert-onnx-to-torch"; }

  StringRef getDescription() const override {
    return "Lower frontend ops to Torch dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  FrontendToTorchLoweringPass() = default;
  FrontendToTorchLoweringPass(const FrontendToTorchLoweringPass &pass)
      : PassWrapper<FrontendToTorchLoweringPass, OperationPass<ModuleOp>>() {}
  FrontendToTorchLoweringPass(bool emitDealloc, bool enableTiling) {
    // Below, need explicit assignment to enable implicit conversion of
    // bool to Option<bool>.
    this->emitDealloc = emitDealloc;
    this->enableTiling = enableTiling;
  }
  FrontendToTorchLoweringPass(int optLevel)
      : FrontendToTorchLoweringPass(false, optLevel >= 3) {}

  void runOnOperation() final;

public:
  // Some ops (RNN ops for example) are lowered to other ONNX ops such as
  // ONNXMatMulOp, ONNXSplitOp, ONNXTransposeOp, etc. These ONNX ops are then
  // lowered into krnl ops in this pass.
  //
  // To write LIT tests for operations that are lowered to other ONNX
  // operations, we do not need to check the final generated krnl code
  // (which is lengthy). It is more convenient to check the intermediate
  // generated code including ONNX ops.
  // We trust the lowering of the other ONNX ops.
  //
  // This flag is used in LIT tests to stop the lowering of the other
  // ONNX ops.
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

void FrontendToTorchLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets
  // for this lowering.
  target.addLegalDialect<Torch::TorchDialect,
      torch::TorchConversion::TorchConversionDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

  // Convert types to legal types for the Torch dialect.
  TorchTypeConverter torchTypeConverter;

  /// Legalize ops
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return torchTypeConverter.isSignatureLegal(op.getFunctionType()) &&
           torchTypeConverter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return torchTypeConverter.isLegal(op);
  });
  target.addLegalOp<ModuleOp>();
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
           isLegalForBranchOpInterfaceTypeConversionPattern(op, torchTypeConverter) ||
           isLegalForReturnOpTypeConversionPattern(op, torchTypeConverter);
  });
  target.addLegalOp<::mlir::UnrealizedConversionCastOp,
      TorchConversion::FromBuiltinTensorOp,
      TorchConversion::ToBuiltinTensorOp>();
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return torchTypeConverter.isLegal(op);
  });

  // Define patterns.
  populateONNXToTorchConversionPattern(
      patterns, torchTypeConverter, &getContext(), enableTiling);

  // Mark materializations as illegal in this pass (since we are finalizing)
  // and add patterns that eliminate them.
  setupFinalization<TorchConversion::ToBuiltinTensorOp,
      TorchConversion::FromBuiltinTensorOp>(target, patterns, torchTypeConverter);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLowerToTorchPass() {
  return std::make_unique<FrontendToTorchLoweringPass>();
}

std::unique_ptr<Pass> createLowerToTorchPass(int optLevel) {
  return std::make_unique<FrontendToTorchLoweringPass>(optLevel);
}
} // namespace onnx_mlir
