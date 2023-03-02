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

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_sha1_ostream.h"

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

  ONNXOpTransformPass() = default;
  ONNXOpTransformPass(const ONNXOpTransformPass &pass)
      : mlir::PassWrapper<ONNXOpTransformPass,
            OperationPass<mlir::ModuleOp>>() {}
  ONNXOpTransformPass(int threshold, bool report, bool targetCPU,
      bool enableSimdDataLayoutOpt) {
    this->onnxOpTransformThreshold = threshold;
    this->onnxOpTransformReport = report;
    this->onnxOpTransformTargetCPU = targetCPU;
    this->onnxOpTransformEnableSimdDataLayout = enableSimdDataLayoutOpt;
  }

  void runOnOperation() final;

private:
  uint64_t createTagForIR(mlir::ModuleOp module) {
    // NOTE: This is slow for real models because they contain large
    // constant tensors that are expensive to print. A workaround is to
    // elide them from printing with --mlir-elide-elementsattrs-if-larger=1
    //
    // TODO: Hash without printing to speed this up, e.g. along the lines of
    // how blocks are hashed in mlir/lib/Transforms/Utils/RegionUtils.cpp
    llvm::raw_sha1_ostream sha1_ostream;
    module->print(sha1_ostream, mlir::OpPrintingFlags());
    std::array<uint8_t, 20> sha1 = sha1_ostream.sha1();
    return *reinterpret_cast<uint64_t *>(sha1.data());
  }
};

void ONNXOpTransformPass::runOnOperation() {
  auto module = getOperation();

  uint64_t currentTag = createTagForIR(module);
  uint64_t previousTag;
  int n = onnxOpTransformThreshold;
  do {
    previousTag = currentTag;
    OpPassManager dynamicPM("builtin.module");
    dynamicPM.addNestedPass<func::FuncOp>(
        onnx_mlir::createDecomposeONNXToONNXPass());
    dynamicPM.addNestedPass<func::FuncOp>(
        onnx_mlir::createDecomposeONNXToONNXWithRankPass());
    dynamicPM.addPass(onnx_mlir::createShapeInferencePass());
    dynamicPM.addPass(mlir::createCanonicalizerPass());
    dynamicPM.addPass(onnx_mlir::createShapeInferencePass());
    // Convolution Optimization currently only for CPU.
    if (onnxOpTransformTargetCPU) {
      dynamicPM.addNestedPass<func::FuncOp>(
          onnx_mlir::createConvOptONNXToONNXPass(
              onnxOpTransformEnableSimdDataLayout));
      dynamicPM.addPass(onnx_mlir::createShapeInferencePass());
    }
    dynamicPM.addNestedPass<func::FuncOp>(
        onnx_mlir::createConstPropONNXToONNXPass());
    if (failed(runPipeline(dynamicPM, module)))
      return signalPassFailure();
    currentTag = createTagForIR(module);
  } while (currentTag != previousTag && --n > 0);
  if (currentTag != previousTag) {
    module->emitWarning()
        << "ONNXOpTransform did not converge after " << onnxOpTransformThreshold
        << "iterations. "
        << "You may set a higher threshold with command option";
  }
  if (onnxOpTransformReport) {
    llvm::outs() << "ONNXOpTransform iterated " << onnxOpTransformThreshold - n
                 << " times, converged "
                 << ((currentTag == previousTag) ? "true" : "false") << "\n";
  }
}

} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createONNXOpTransformPass() {
  return std::make_unique<ONNXOpTransformPass>();
}

std::unique_ptr<mlir::Pass> onnx_mlir::createONNXOpTransformPass(
    int threshold, bool report, bool targetCPU, bool enableSimdDataLayoutOpt) {
  return std::make_unique<ONNXOpTransformPass>(
      threshold, report, targetCPU, enableSimdDataLayoutOpt);
}
