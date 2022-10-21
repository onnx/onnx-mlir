/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- InstrumentPass.cpp - Instrumentation ---------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass that inserts instrumentation.
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
// #include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/*!
 * This pass insert KrnlInstrumentOp before and after each ops in specified
 * dialect
 */

class InstrumentPass
    : public mlir::PassWrapper<InstrumentPass, OperationPass<func::FuncOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InstrumentPass)

  Option<std::string> instrumentDialects{*this, "instrument-dialects",
      llvm::cl::desc("Specify dialect to be instrumented\n"
                     "\"NONE\" or \"\" for no instrument\n"
                     "\"ALL\" for all dialects. \n"
                     "\"dialect1 dialect2 ...\" for the specified dialect."),
      llvm::cl::init("")};

  Option<std::string> instrumentOps{*this, "instrument-ops",
      llvm::cl::desc("Specify ops to be instrumented\n"
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

  InstrumentPass() = default;
  InstrumentPass(const InstrumentPass &pass)
      : mlir::PassWrapper<InstrumentPass, OperationPass<func::FuncOp>>() {}
  InstrumentPass(StringRef dialects, StringRef ops, unsigned actions) {
    this->instrumentDialects = dialects.str();
    this->instrumentOps = ops.str();
    this->instrumentBefore = actions & (1 << onnx_mlir::InstrumentBeforeOp);
    this->instrumentAfter = actions & (1 << onnx_mlir::InstrumentAfterOp);
    this->reportTime = actions & (1 << onnx_mlir::InstrumentReportTime);
    this->reportMemory = actions & (1 << onnx_mlir::InstrumentReportMemory);
  }

private:
  bool allOpsAllowed;
  std::set<std::string> allowedOps;

public:
  StringRef getArgument() const override { return "instrument-ops"; }

  StringRef getDescription() const override {
    return "instrument on ops of specific dialect.";
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
    if (instrumentOps == "" || instrumentOps == "NONE")
      return;
    init(instrumentOps);

    // Iterate on the operations nested in this function
    getOperation().walk([&](mlir::Operation *op) {
      if (StringRef(instrumentDialects)
              .equals_insensitive(op->getDialect()->getNamespace())) {
        // Skip the dialect name
        const char *opName =
            op->getName().getStringRef().data() + instrumentDialects.size();
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
std::unique_ptr<mlir::Pass> onnx_mlir::createInstrumentPass() {
  return std::make_unique<InstrumentPass>();
}

std::unique_ptr<mlir::Pass> onnx_mlir::createInstrumentPass(
    StringRef dialects, StringRef ops, unsigned actions) {
  return std::make_unique<InstrumentPass>(dialects, ops, actions);
}
