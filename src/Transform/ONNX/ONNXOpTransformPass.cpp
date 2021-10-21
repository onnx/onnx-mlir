/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- InstrumentONNXPass.cpp - Instrumentation ---------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass that inserts instrumentation
// for ONNX ops.
//
//===----------------------------------------------------------------------===//

#include <set>
#include <fstream>
#include <iostream>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ToolOutputFile.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/OMOptions.hpp"

using namespace mlir;

namespace {

/*!
 * This pass insert KrnlInstrumentOp before and after each ONNX ops
 */

class ONNXGraphOptimizePass : public mlir::PassWrapper<ONNXGraphOptimizePass,
                                  OperationPass<mlir::ModuleOp>> {

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
    char tempFile[] = "/tmp/onnxpassdumpXXXXXX";
    mkstemp(tempFile);
    outputCode(module, tempFile);
    uint64_t r = hashFile(tempFile);
    remove(tempFile);
    return r;
  }

public:
  ONNXGraphOptimizePass() = default;
  ONNXGraphOptimizePass(const ONNXGraphOptimizePass &pass) {}
  ONNXGraphOptimizePass(int threshold) {
    this->optimizeThreshold = threshold;
  }

  Option<int> optimizeThreshold{*this, "onnx-graph-optimize-threshold",
      llvm::cl::desc("max iteration for graph optimization passes."),
      llvm::cl::init(3)};
  void runOnOperation() override {
    auto module = getOperation();

    uint64_t currentTag = createTagForIR(module);
    uint64_t previousTag;

    int n = 5;
    do {
      previousTag = currentTag;
      printf("tag#%d %lu\n", n, currentTag);
      OpPassManager dynamicPM("builtin.module");
      dynamicPM.addNestedPass<FuncOp>(mlir::createDecomposeONNXToONNXPass());
      dynamicPM.addPass(mlir::createShapeInferencePass());
      dynamicPM.addPass(mlir::createCanonicalizerPass());
      dynamicPM.addPass(mlir::createShapeInferencePass());
      dynamicPM.addNestedPass<FuncOp>(mlir::createConstPropONNXToONNXPass());
      dynamicPM.addPass(mlir::createSymbolDCEPass());
      if (failed(runPipeline(dynamicPM, module)))
        return signalPassFailure();
      currentTag = createTagForIR(module);
    } while (currentTag != previousTag && --n > 0);
  }
};

} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> mlir::createONNXGraphOptimizePass() {
  return std::make_unique<ONNXGraphOptimizePass>();
}

std::unique_ptr<mlir::Pass> mlir::createONNXGraphOptimizePass(int threshold) {
  return std::make_unique<ONNXGraphOptimizePass>(threshold);
}
