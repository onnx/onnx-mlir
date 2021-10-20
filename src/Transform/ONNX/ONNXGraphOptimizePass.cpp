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

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

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

class ONNXGraphOptimizePass
    : public mlir::PassWrapper<ONNXGraphOptimizePass, OperationPass<mlir::ModuleOp>> {

private:
  
public:
  void runOnOperation() override {
    auto module = getOperation();
    OpPassManager dynamicPM("builtin.module");
    dynamicPM.addNestedPass<FuncOp>(mlir::createDecomposeONNXToONNXPass());
    dynamicPM.addPass(mlir::createShapeInferencePass());
    dynamicPM.addPass(mlir::createCanonicalizerPass());
    dynamicPM.addPass(mlir::createShapeInferencePass());
    dynamicPM.addNestedPass<FuncOp>(mlir::createConstPropONNXToONNXPass());
    dynamicPM.addPass(mlir::createSymbolDCEPass());
    if (failed(runPipeline(dynamicPM, module)))
      return signalPassFailure();
  }
};

} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> mlir::createONNXGraphOptimizePass() {
  return std::make_unique<ONNXGraphOptimizePass>();
}
