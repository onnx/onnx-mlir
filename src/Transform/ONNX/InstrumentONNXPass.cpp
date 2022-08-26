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

#include "onnx-mlir/Compiler/OMCompilerTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

class InstrumentONNXPass : public mlir::PassWrapper<InstrumentONNXPass,
                               OperationPass<func::FuncOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InstrumentONNXPass)

  Option<std::string> instrumentONNXOps{*this, "instrument-onnx-ops",
      llvm::cl::desc("Specify onnx ops to be instrumented\n"
                     "\"NONE\" or \"\" for no instrument\n"
                     "\"ALL\" for all ops. \n"
                     "\"op1 op2 ...\" for the specified ops."),
      llvm::cl::init("")};

  Option<bool> instrumentBefore{*this, "instrument-before",
      llvm::cl::desc("insert instrument before op"), llvm::cl::init(false)};

  Option<bool> instrumentAfter{*this, "instrument-after",
      llvm::cl::desc("insert instrument after op"), llvm::cl::init(false)};

  Option<bool> reportTime{*this, "report-time",
      llvm::cl::desc("instrument runtime reports time usage"),
      llvm::cl::init(false)};

  Option<bool> reportMemory{*this, "report-memory",
      llvm::cl::desc("instrument runtime reports memory usage"),
      llvm::cl::init(false)};

  InstrumentONNXPass() = default;
  InstrumentONNXPass(const InstrumentONNXPass &pass)
      : mlir::PassWrapper<InstrumentONNXPass, OperationPass<func::FuncOp>>() {}
  InstrumentONNXPass(StringRef ops, unsigned actions) {
    this->instrumentONNXOps = ops.str();
    this->instrumentBefore = actions & (1 << onnx_mlir::InstrumentBeforeOp);
    this->instrumentAfter = actions & (1 << onnx_mlir::InstrumentAfterOp);
    this->reportTime = actions & (1 << onnx_mlir::InstrumentReportTime);
    this->reportMemory = actions & (1 << onnx_mlir::InstrumentReportMemory);
  }

private:
  bool allOpsAllowed;
  std::set<std::string> allowedOps;

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
  }

  // merge all action options into a bitset
  // used to create tags for instrumentation ops
  int actions() const {
    int tag = 0;
    if (instrumentBefore)
      tag |= 1 << onnx_mlir::InstrumentBeforeOp;
    if (instrumentAfter)
      tag |= 1 << onnx_mlir::InstrumentAfterOp;
    if (reportTime)
      tag |= 1 << onnx_mlir::InstrumentReportTime;
    if (reportMemory)
      tag |= 1 << onnx_mlir::InstrumentReportMemory;
    return tag;
  }

  int beforeTag() const {
    return actions() & (~(1 << onnx_mlir::InstrumentAfterOp));
  }
  int afterTag() const {
    return actions() & (~(1 << onnx_mlir::InstrumentBeforeOp));
  }

  void runOnOperation() override {
    if (instrumentONNXOps == "" || instrumentONNXOps == "NONE")
      return;
    init(instrumentONNXOps);

    // Iterate on the operations nested in this function
    getOperation().walk([&](mlir::Operation *op) {
      if (isa<mlir::ONNXDialect>(op->getDialect())) {
        // Skip the prefix "onnx." of onnx op name
        const char *opName = op->getName().getStringRef().data() + 5;
        if (!allOpsAllowed && allowedOps.find(opName) == allowedOps.end())
          return;

        Location loc = op->getLoc();
        OpBuilder opBuilder(op);
        if (instrumentBefore)
          opBuilder.create<mlir::KrnlInstrumentOp>(loc, op, beforeTag());

        // Can not insert after Op (e.g. ONNXReturnOP) with IsTerminator Trait
        if (instrumentAfter && !op->hasTrait<OpTrait::IsTerminator>()) {
          opBuilder.setInsertionPointAfter(op);
          opBuilder.create<mlir::KrnlInstrumentOp>(loc, op, afterTag());
        }
      }
    });
  }
};
} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createInstrumentONNXPass() {
  return std::make_unique<InstrumentONNXPass>();
}

std::unique_ptr<mlir::Pass> onnx_mlir::createInstrumentONNXPass(
    StringRef ops, unsigned actions) {
  return std::make_unique<InstrumentONNXPass>(ops, actions);
}
