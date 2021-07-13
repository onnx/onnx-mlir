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
// for ops.
//
//===----------------------------------------------------------------------===//

#include <regex>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/*!
 *  FunctionPass that performs shape inference by iterating over a list of
 *  candidate operations and propagating the shape information until the list
 *  of operations is empty [credit MLIR authors].
 *
 * Shape inference proceeds recursively, starting with the entry point function
 * corresponding to the main computation graph. This is because sometimes an
 * operation is associated with a different (sub) computation graph in the forms
 * of mlir functions, and the operation's output shape and type depends on the
 * shape and type of that (sub) graph outputs. In such scenarios, operations can
 * initiate shape inference on its dependent (sub) graph, and resume infering
 * its output shape only after shape inference completes for the associated
 * (sub) graph.
 *
 * In the absence of a main computation graph, we will treat every mlir
 * function as a main computation graph; this is mostly just for testing
 * purposes.
 */
class InstrumentONNXPass : public mlir::PassWrapper<InstrumentONNXPass,
                               OperationPass<mlir::ModuleOp>> {
private:
  bool instrumentEnabled;

public:
  InstrumentONNXPass(bool instrumentEnabled_)
      : instrumentEnabled(instrumentEnabled_) {}

  void runOnOperation() override {
    if (!instrumentEnabled)
	    return;

    auto module = getOperation();
    auto result = module.walk([&](FuncOp funcOp) -> WalkResult {
      return runInstrumentONNXOn(funcOp);
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }

  static LogicalResult runInstrumentONNXOnRegion(mlir::Region &r) {
    // Iterate on the operations 
    for (Operation &op : r.getOps()) {
      if (isa<mlir::ONNXOpsDialect>(op.getDialect())) {
	      Location loc = op.getLoc();
	      OpBuilder opBuilder(&op);
	      opBuilder.create<mlir::KrnlInstrumentOp>(loc, &op, 0);
	      opBuilder.setInsertionPointAfter(&op);
	      opBuilder.create<mlir::KrnlInstrumentOp>(loc, &op, 1);

      }
    }
    return success();
  }

  static LogicalResult runInstrumentONNXOn(mlir::FuncOp f) {
    auto &funcBody = f.getBody();
    if (failed(runInstrumentONNXOnRegion(funcBody)))
      return failure();

    return success();
  }

};
} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> mlir::createInstrumentONNXPass(
    bool enabled) {
  return std::make_unique<InstrumentONNXPass>(enabled);
}
