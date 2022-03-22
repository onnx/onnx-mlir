/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToTorch.cpp - ONNX dialects to Torch lowering -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to a combination of
// Torch IR and standard operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

void populateONNXToTorchConversionPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableTiling) {
  
    populateLoweringONNXToTorchConvOpPattern (patterns, typeConverter, ctx);
    populateLoweringONNXToTorchConstantPadNdOpPattern (patterns, typeConverter, ctx);
    populateLoweringONNXToTorchLeakyReluOpPattern (patterns, typeConverter, ctx);
    populateLoweringONNXToTorchMaxPoolSingleOutOpPattern (patterns, typeConverter, ctx);
    populateLoweringONNXToTorchConstOpPattern (patterns, typeConverter, ctx);
}

//===----------------------------------------------------------------------===//
// Frontend to Torch Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Torch loops of the ONNX operations.
namespace {
struct FrontendToTorchLoweringPass
    : public PassWrapper<FrontendToTorchLoweringPass, OperationPass<::mlir::ModuleOp>> {

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
    // Below, need explicit assignment to enable implicit conversion of bool to
    // Option<bool>.
    this->emitDealloc = emitDealloc;
    this->enableTiling = enableTiling;
  }
  FrontendToTorchLoweringPass(int optLevel)
      : FrontendToTorchLoweringPass( false, optLevel >= 3) {}

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

void FrontendToTorchLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target
      .addLegalDialect<Torch::TorchDialect>();
  target
      .addLegalDialect<torch::TorchConversion::TorchConversionDialect>();

  // Needed to support unsigned int computations. To be removed if we use a
  // scheme that does not rely on the UnrealizedConversionCastOp.
  target.addLegalOp<::mlir::UnrealizedConversionCastOp>();

  // If `emitDealloc` is turned off, make sure we don't have buffer deallocation
  // at this level. Will use MLIR buffer-deallocation for this purpose instead.
  if (!emitDealloc)
    target.addIllegalOp<mlir::memref::DeallocOp>();


  if (emitIntermediateIR) {
    // Only used for writing LIT tests for ONNX operations that are lowered to
    // other ONNX operations. The following operations are prevented from being
    // lowered further. See the comment in the declaration of
    // 'emitIntermediateIR' for more details.

    #if 0
    target.addLegalOp<ONNXMatMulOp>();
    target.addLegalOp<ONNXReshapeOp>();
    target.addLegalOp<ONNXSplitV11Op>();
    target.addLegalOp<ONNXSqueezeV11Op>();
    target.addLegalOp<ONNXTransposeOp>();
    #endif
  }

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

  // Convert types to legal types for the Krnl dialect.
  TorchTypeConverter torchTypeConverter;
  // Define patterns.
  populateONNXToTorchConversionPattern(
      patterns, torchTypeConverter, &getContext(), enableTiling);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createLowerToTorchPass() {
  return std::make_unique<FrontendToTorchLoweringPass>();
}

std::unique_ptr<Pass> mlir::createLowerToTorchPass(int optLevel) {
  return std::make_unique<FrontendToTorchLoweringPass>(optLevel);
}

