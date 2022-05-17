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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ToolOutputFile.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;

namespace {

/*!
 * This pass insert KrnlInstrumentOp before and after each ONNX ops
 */

struct ONNXOpTransformPass : public mlir::PassWrapper<ONNXOpTransformPass,
                                 OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXOpTransformPass)

  StringRef getArgument() const override { return "onnx-op-transform"; }

  StringRef getDescription() const override {
    return "Invoke passes iteratively that transform ONNX operation.";
  }

  Option<int> onnxOpTransformThreshold{*this, "onnx-op-transform-threshold",
      llvm::cl::desc("max iteration for op transform passes."),
      llvm::cl::init(3)};

  Option<bool> onnxOpTransformReport{*this, "onnx-op-transform-report",
      llvm::cl::desc("Report diagnostic info for op transform passes."),
      llvm::cl::init(false)};

  ONNXOpTransformPass() = default;
  ONNXOpTransformPass(const ONNXOpTransformPass &pass)
      : mlir::PassWrapper<ONNXOpTransformPass,
            OperationPass<mlir::ModuleOp>>() {}
  ONNXOpTransformPass(int threshold, bool report) {
    this->onnxOpTransformThreshold = threshold;
    this->onnxOpTransformReport = report;
  }

  void runOnOperation() final;

private:
  void outputCode(mlir::ModuleOp module, std::string filename) {
    mlir::OpPrintingFlags flags;

    std::string errorMessage;
    auto output = mlir::openOutputFile(filename, &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
      exit(1);
    }

    module->print(output->os(), flags);
    output->keep();

    // Code may be needed with flag control for debugging in future
    // if (printIR)
    // module->print(llvm::outs(), flags);
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

  uint64_t createTagForIR(mlir::ModuleOp module) {
    char tempFile[64];
#ifdef _WIN32
    strcpy(tempFile, "onnxtempdumpXXXXXX");
    _mktemp(tempFile);
#else
    strcpy(tempFile, "/tmp/onnxtempdumpXXXXXX");
    (void)mkstemp(tempFile);
#endif
    outputCode(module, tempFile);
    uint64_t r = hashFile(tempFile);
    remove(tempFile);
    return r;
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
    dynamicPM.addPass(onnx_mlir::createShapeInferencePass());
    dynamicPM.addPass(mlir::createCanonicalizerPass());
    dynamicPM.addPass(onnx_mlir::createShapeInferencePass());
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
    int threshold, bool report) {
  return std::make_unique<ONNXOpTransformPass>(threshold, report);
}
