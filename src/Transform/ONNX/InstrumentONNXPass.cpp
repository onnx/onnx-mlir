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

using namespace mlir;

namespace {

/*!
 * This pass insert KrnlInstrumentOp before and after each ONNX ops
 */

// TO FIX: this option should be put in the ONNXMlirOptions category
// However, currently ONNXMlirOptions is defined in MainUtils.cpp
// If this option is defined there, the enum type InsertPostion will be moved
// out We may need to work a little more to structure files for command line
// options Another related issue is that the enum type is used by onnx-mlir
// Runtime too.

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
        clEnumVal(InstrumentReportMemory,
            "instrument runtime reports memory usage")));

class InstrumentONNXPass
    : public mlir::PassWrapper<InstrumentONNXPass, FunctionPass> {

private:
  bool allOpsAllowed;
  std::set<std::string> allowedOps;
  unsigned runtimeActions;

public:
  InstrumentONNXPass(std::string allowedOps_) {
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

  void runOnFunction() override {
    auto function = getFunction();
    auto &funcBody = function.getBody();

    // Iterate on the operations
    for (Operation &op : funcBody.getOps()) {
      if (isa<mlir::ONNXOpsDialect>(op.getDialect())) {
        // Skip the prefix "onnx." of onnx op name
        const char *opName = op.getName().getStringRef().data() + 5;
        if (!allOpsAllowed && allowedOps.find(opName) == allowedOps.end())
          continue;

        Location loc = op.getLoc();
        OpBuilder opBuilder(&op);
        if (InstrumentControlBits.isSet(InstrumentBeforeOp)) {
          uint64_t tag =
              runtimeActions & (~(1 << static_cast<int>(InstrumentBeforeOp)));
          opBuilder.create<mlir::KrnlInstrumentOp>(loc, &op, tag);
        }
        if (InstrumentControlBits.isSet(InstrumentAfterOp)) {
          opBuilder.setInsertionPointAfter(&op);
          uint64_t tag =
              runtimeActions & (~(1 << static_cast<int>(InstrumentAfterOp)));
          opBuilder.create<mlir::KrnlInstrumentOp>(loc, &op, tag);
        }
      }
    }
  }
};
} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> mlir::createInstrumentONNXPass(
    std::string allowedOps) {
  return std::make_unique<InstrumentONNXPass>(allowedOps);
}
