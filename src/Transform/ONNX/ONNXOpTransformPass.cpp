/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ONNXOpTransformPass.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

struct ONNXOpTransformPass : public mlir::PassWrapper<ONNXOpTransformPass,
                                 OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXOpTransformPass)

  StringRef getArgument() const override { return "onnx-op-transform"; }

  StringRef getDescription() const override {
    return "Invoke passes iteratively that transform ONNX operation.";
  }

  Option<int> onnxOpTransformThreshold{*this, "onnx-op-transform-threshold",
      llvm::cl::desc("max iteration for op transform passes."),
      llvm::cl::init(10)};
  Option<bool> onnxOpTransformReport{*this, "onnx-op-transform-report",
      llvm::cl::desc("Report diagnostic info for op transform passes."),
      llvm::cl::init(false)};
  Option<bool> onnxOpTransformTargetCPU{*this, "onnx-op-transform-target-cpu",
      llvm::cl::desc("Target CPU op transform passes."), llvm::cl::init(true)};
  Option<bool> onnxOpTransformEnableSimdDataLayout{*this,
      "onnx-op-transform-simd-data-layout",
      llvm::cl::desc("Enable SIMD data layout opt in op transform passes."),
      llvm::cl::init(false)};
  Option<bool> enableConvOptPass{*this, "enable-conv-opt-pass",
      llvm::cl::desc("Enable the ConvOptPass. Default is true."),
      llvm::cl::init(true)};
  Option<bool> enableRecomposeOptPass{*this, "enable-recompose-opt-pass",
      llvm::cl::desc("Enable the RecomposeOptPass. Default is true."),
      llvm::cl::init(true)};

  ONNXOpTransformPass() = default;
  ONNXOpTransformPass(const ONNXOpTransformPass &pass)
      : mlir::PassWrapper<ONNXOpTransformPass,
            OperationPass<mlir::ModuleOp>>() {}
  ONNXOpTransformPass(int threshold, bool report, bool targetCPU,
      bool enableSimdDataLayoutOpt, bool enableConvOptPass,
      bool enableRecomposeOptPass) {
    this->onnxOpTransformThreshold = threshold;
    this->onnxOpTransformReport = report;
    this->onnxOpTransformTargetCPU = targetCPU;
    this->onnxOpTransformEnableSimdDataLayout = enableSimdDataLayoutOpt;
    this->enableConvOptPass = enableConvOptPass;
    this->enableRecomposeOptPass = enableRecomposeOptPass;
  }

  void runOnOperation() final;
};

void ONNXOpTransformPass::runOnOperation() {
  auto module = getOperation();

  assert(onnxOpTransformThreshold > 0);
  int n = onnxOpTransformThreshold;
  OperationFingerPrint before(module);
  do {
    OpPassManager dynamicPM("builtin.module");
    dynamicPM.addNestedPass<func::FuncOp>(
        onnx_mlir::createDecomposeONNXToONNXPass());
    if (enableRecomposeOptPass)
      dynamicPM.addNestedPass<func::FuncOp>(
          onnx_mlir::createRecomposeONNXToONNXPass());
    dynamicPM.addNestedPass<func::FuncOp>(
        onnx_mlir::createShapeInferencePass());
    dynamicPM.addPass(mlir::createCanonicalizerPass());
    dynamicPM.addNestedPass<func::FuncOp>(
        onnx_mlir::createShapeInferencePass());
    // Convolution Optimization currently only for CPU.
    if (onnxOpTransformTargetCPU && enableConvOptPass) {
      dynamicPM.addNestedPass<func::FuncOp>(
          onnx_mlir::createConvOptONNXToONNXPass(
              onnxOpTransformEnableSimdDataLayout));
      dynamicPM.addNestedPass<func::FuncOp>(
          onnx_mlir::createShapeInferencePass());
    }
    dynamicPM.addNestedPass<func::FuncOp>(
        onnx_mlir::createConstPropONNXToONNXPass());
    if (failed(runPipeline(dynamicPM, module)))
      return signalPassFailure();
    OperationFingerPrint after(module);
    if (after == before)
      break;
    before = after;
  } while (--n > 0);
  if (n == 0) {
    module->emitWarning()
        << "ONNXOpTransform did not converge after " << onnxOpTransformThreshold
        << "iterations. "
        << "You may set a higher threshold with command option";
  }
  if (onnxOpTransformReport) {
    llvm::outs() << "ONNXOpTransform iterated " << onnxOpTransformThreshold - n
                 << " times, converged " << (n > 0 ? "true" : "false") << "\n";
  }
}

} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createONNXOpTransformPass() {
  return std::make_unique<ONNXOpTransformPass>();
}

std::unique_ptr<mlir::Pass> onnx_mlir::createONNXOpTransformPass(int threshold,
    bool report, bool targetCPU, bool enableSimdDataLayoutOpt,
    bool enableConvOptPass, bool enableRecomposeOptPass) {
  return std::make_unique<ONNXOpTransformPass>(threshold, report, targetCPU,
      enableSimdDataLayoutOpt, enableConvOptPass, enableRecomposeOptPass);
}
