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

// Strong-typed enum is NOT used because the value will be static_cast to int
enum InstrumentActions {
  InstrumentBeforeOp,
  InstrumentAfterOp,
  InstrumentReportTime,
  InstrumentReportMemory
};

// Inherited issue: no default value support for cl::bits
llvm::cl::bits<InstrumentActions> InstrumentControlBits(
    llvm::cl::desc("Specify what instrumentation actions at runtime:"),
    llvm::cl::values(
        clEnumVal(InstrumentBeforeOp, "insert instrument before op"),
        clEnumVal(InstrumentAfterOp, "insert instrument after op"),
        clEnumVal(
            InstrumentReportTime, "instrument runtime reports time usage"),
        clEnumVal(
            InstrumentReportMemory, "instrument runtime reports memory usage")),
    llvm::cl::cat(OMPassOptions));

class InstrumentONNXPass
    : public mlir::PassWrapper<InstrumentONNXPass, OperationPass<FuncOp>> {

private:
  bool allOpsAllowed;
  std::set<std::string> allowedOps;
  unsigned runtimeActions;

public:
  StringRef getArgument() const override { return "instrument-onnx"; }

  StringRef getDescription() const override {
    return "instrument on onnx ops.";
  }

  void init(std::string allowedOps_) {
    if (allowedOps_ == "ALL") {
      allOpsAllowed = true;
    } else {
      allOpsAllowed = false;
      std::stringstream ss(allowedOps_);
      std::istream_iterator<std::string> begin(ss);
      std::istream_iterator<std::string> end;
      allowedOps = std::set<std::string>(begin, end);
    }
    runtimeActions = InstrumentControlBits.getBits();
  };

  void runOnOperation() override {
    if (instrumentONNXOps == "" || instrumentONNXOps == "NONE")
      return;
    init(instrumentONNXOps);

    // Iterate on the operations nested in this function
    getOperation().walk([&](mlir::Operation *op) {
      if (isa<mlir::ONNXOpsDialect>(op->getDialect())) {
        // Skip the prefix "onnx." of onnx op name
        const char *opName = op->getName().getStringRef().data() + 5;
        if (!allOpsAllowed && allowedOps.find(opName) == allowedOps.end())
          return;

        Location loc = op->getLoc();
        OpBuilder opBuilder(op);
        if (InstrumentControlBits.isSet(InstrumentBeforeOp)) {
          uint64_t tag =
              runtimeActions & (~(1 << static_cast<int>(InstrumentAfterOp)));
          opBuilder.create<mlir::KrnlInstrumentOp>(loc, op, tag);
        }

        // Can not insert after Op (e.g. ONNXReturnOP) with IsTerminator Trait
        if (InstrumentControlBits.isSet(InstrumentAfterOp) &&
            !op->hasTrait<OpTrait::IsTerminator>()) {
          opBuilder.setInsertionPointAfter(op);
          uint64_t tag =
              runtimeActions & (~(1 << static_cast<int>(InstrumentBeforeOp)));
          opBuilder.create<mlir::KrnlInstrumentOp>(loc, op, tag);
        }
      }
    });
  }
};
} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> mlir::createInstrumentONNXPass() {
  return std::make_unique<InstrumentONNXPass>();
}
