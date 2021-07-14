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

using namespace mlir;

namespace {

/*!
 * This pass insert KrnlInstrumentOp before and after each ONNX ops
 */

class InstrumentONNXPass
    : public mlir::PassWrapper<InstrumentONNXPass, FunctionPass> {

private :
  std::string opsString;

public:
  InstrumentONNXPass(std::string allowedOps) opsString(allowedOps) {};

  void runOnFunction() override {
    auto function = getFunction();
    auto &funcBody = function.getBody();

    // Iterate on the operations
    for (Operation &op : funcBody.getOps()) {
      if (isa<mlir::ONNXOpsDialect>(op.getDialect())) {
        Location loc = op.getLoc();
        OpBuilder opBuilder(&op);
        opBuilder.create<mlir::KrnlInstrumentOp>(loc, &op, 0);
        opBuilder.setInsertionPointAfter(&op);
        opBuilder.create<mlir::KrnlInstrumentOp>(loc, &op, 1);
      }
    }
  }
};
} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> mlir::createInstrumentONNXPass() {
  return std::make_unique<InstrumentONNXPass>();
}
