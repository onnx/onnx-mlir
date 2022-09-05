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

#include <fstream>
#include <iostream>
#include <set>

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ToolOutputFile.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#ifdef _WIN32
#include <io.h>
#endif

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

  ONNXOpTransformPass() = default;
  ONNXOpTransformPass(const ONNXOpTransformPass &pass)
      : mlir::PassWrapper<ONNXOpTransformPass,
            OperationPass<mlir::ModuleOp>>() {}
  ONNXOpTransformPass(int threshold, bool report, bool targetCPU) {
    this->onnxOpTransformThreshold = threshold;
    this->onnxOpTransformReport = report;
    this->onnxOpTransformTargetCPU = targetCPU;
  }

  void runOnOperation() final;

private:
  LogicalResult outputCode(mlir::ModuleOp module, std::string filename) {
    mlir::OpPrintingFlags flags;

    std::string errorMessage;
    auto output = mlir::openOutputFile(filename, &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }

    module->print(output->os(), flags);
    output->keep();

    // Code may be needed with flag control for debugging in future
    // if (printIR)
    // module->print(llvm::outs(), flags);
    return success();
  }

  uint64_t hashFile(std::string filename) {
    std::ifstream t(filename);
    std::stringstream buffer;
    buffer << t.rdbuf();

    // Copied from current version of llvm Support/MD5.h
    // return md5.MD5Hash(buffer.str());
    llvm::MD5 Hash;
    Hash.update(buffer.str());
    llvm::MD5::MD5Result Result;
    Hash.final(Result);
    return Result.low();
  }

  LogicalResult createTagForIR(mlir::ModuleOp module, uint64_t *tag) {
    llvm::SmallString<64> tempFile;
    // LLVM provides functionality for securely creating randomly named files in
    // the appropriate tmp directory (it works on any platform). Use it here
    // rather than directly calling OS-specific methods.
    if (auto ec =
            llvm::sys::fs::createTemporaryFile("onnxtempdump", "", tempFile)) {
      llvm::errs() << ec.message() << "\n";
      return failure();
    }

    // This will remove the file when it goes out of scope.
    llvm::FileRemover tempFileRemover(tempFile);

    if (failed(outputCode(module, std::string(tempFile))))
      return failure();

    *tag = hashFile(std::string(tempFile));
    return success();
  }
};

void ONNXOpTransformPass::runOnOperation() {
  auto module = getOperation();

  uint64_t currentTag;
  uint64_t previousTag;

  if (failed(createTagForIR(module, &currentTag)))
    return signalPassFailure();

  int n = onnxOpTransformThreshold;
  bool targetCPU = onnxOpTransformTargetCPU;
  do {
    previousTag = currentTag;
    OpPassManager dynamicPM("builtin.module");
    dynamicPM.addNestedPass<func::FuncOp>(
        onnx_mlir::createDecomposeONNXToONNXPass());
    dynamicPM.addPass(onnx_mlir::createShapeInferencePass());
    dynamicPM.addPass(mlir::createCanonicalizerPass());
    dynamicPM.addPass(onnx_mlir::createShapeInferencePass());
    // Convolution Optimization currently only for CPU.
    if (targetCPU) {
      dynamicPM.addNestedPass<func::FuncOp>(
          onnx_mlir::createConvOptONNXToONNXPass());
      dynamicPM.addPass(onnx_mlir::createShapeInferencePass());
    }
    dynamicPM.addNestedPass<func::FuncOp>(
        onnx_mlir::createConstPropONNXToONNXPass());
    if (failed(runPipeline(dynamicPM, module)))
      return signalPassFailure();
    if (failed(createTagForIR(module, &currentTag)))
      return signalPassFailure();
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
    int threshold, bool report, bool targetCPU) {
  return std::make_unique<ONNXOpTransformPass>(threshold, report, targetCPU);
}
