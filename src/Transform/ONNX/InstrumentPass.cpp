/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- InstrumentPass.cpp - Instrumentation ---------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass that inserts instrumentation.
//
//===----------------------------------------------------------------------===//

#include <regex>
#include <set>
#include <string>

#include "onnx-mlir/Compiler/OMCompilerRuntimeTypes.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Compiler/OptionUtils.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

/*!
 * This pass insert KrnlInstrumentOp before and after each ops
 */

class InstrumentPass
    : public mlir::PassWrapper<InstrumentPass, OperationPass<func::FuncOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InstrumentPass)

  Option<std::string> instrumentOps{*this, "instrument-ops",
      llvm::cl::desc("Specify regex for ops to be instrumented:\n"
                     "\"NONE\" or \"\" for no instrument,\n"
                     "\"regex1,regex2, ...\" for the specified ops.\n"
                     "e.g. \"onnx.,zhigh.\" for onnx and zhigh ops.\n"
                     "e.g. \"onnx.Conv\" for onnx Conv ops.\n"),
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

  InstrumentPass() : allowedOps(/*emptyIsNone*/ true){};
  InstrumentPass(const InstrumentPass &pass)
      : mlir::PassWrapper<InstrumentPass, OperationPass<func::FuncOp>>(),
        allowedOps(/*emptyIsNone*/ true) {}
  InstrumentPass(const std::string &ops, unsigned actions)
      : allowedOps(/*emptyIsNone*/ true) {
    this->instrumentOps = ops;
    unsigned long long tag = actions;
    this->instrumentBefore = IS_INSTRUMENT_BEFORE_OP(tag);
    this->instrumentAfter = IS_INSTRUMENT_AFTER_OP(tag);
    this->reportTime = IS_INSTRUMENT_REPORT_TIME(tag);
    this->reportMemory = IS_INSTRUMENT_REPORT_MEMORY(tag);
  }

private:
  EnableByRegexOption allowedOps;

public:
  StringRef getArgument() const override { return "instrument"; }

  StringRef getDescription() const override { return "instrument on ops."; }

  // merge all action options into a bitset
  // used to create tags for instrumentation ops
  uint64_t actions() const {
    int64_t tag;

    INIT_INSTRUMENT(tag);
    if (instrumentBefore)
      SET_INSTRUMENT_BEFORE_OP(tag);
    if (instrumentAfter)
      SET_INSTRUMENT_AFTER_OP(tag);
    if (reportTime)
      SET_INSTRUMENT_REPORT_TIME(tag);
    if (reportMemory)
      SET_INSTRUMENT_REPORT_MEMORY(tag);
    return tag;
  }

  uint64_t beforeTag() const {
    int64_t tag = actions();
    CLEAR_INSTRUMENT_AFTER_OP(tag);
    return tag;
  }

  uint64_t afterTag() const {
    int64_t tag = actions();
    CLEAR_INSTRUMENT_BEFORE_OP(tag);
    return tag;
  }

  void runOnOperation() override {
    if (instrumentOps == "" || instrumentOps == "NONE")
      return;
    allowedOps.setRegexString(instrumentOps);
    bool hasInitializedRuntime = false;

    // Iterate on the operations nested in this function
    getOperation().walk([&](mlir::Operation *op) -> WalkResult {
      // Do not profile operations that return a None value (e.g. onnx.NoValue).
      // Somehow such none-returned operations cause messy output, For example,
      // with --profile-ir=ZHigh for the mnist-12 model, it mixed version error
      // with profiling info.
      // ```
      // #  0) before zlow.stickModel is running on hardware that is not
      // compatible with the zDNN library that this model was compiled for
      // (version num %llu.%llu.%llu). Please check that the model is running on
      // hardware with an integrated accelerator for AI (z16 +) that supports
      // the required zDNN library version.
      // ```
      if (op->getNumResults() == 1 && isa<NoneType>(op->getResult(0).getType()))
        return WalkResult::advance();
      std::string opName = op->getName().getStringRef().str();
      if (allowedOps.isEnabled(opName)) {
        Location loc = op->getLoc();
        OpBuilder opBuilder(op);
        if (instrumentBefore) {
          uint64_t tag = beforeTag();
          if (!hasInitializedRuntime) {
            SET_INSTRUMENT_INIT(tag);
            hasInitializedRuntime = true;
          }
          opBuilder.create<mlir::KrnlInstrumentOp>(loc, op, tag);
        }

        // Can not insert after Op (e.g. ONNXYieldOP) with IsTerminator Trait
        if (instrumentAfter && !op->hasTrait<OpTrait::IsTerminator>()) {
          opBuilder.setInsertionPointAfter(op);
          uint64_t tag = afterTag();
          if (!hasInitializedRuntime) {
            SET_INSTRUMENT_INIT(tag);
            hasInitializedRuntime = true;
          }
          opBuilder.create<mlir::KrnlInstrumentOp>(loc, op, tag);
        }
      }
      return WalkResult::advance();
    });
  }
};
} // namespace onnx_mlir

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createInstrumentPass() {
  return std::make_unique<InstrumentPass>();
}

std::unique_ptr<mlir::Pass> onnx_mlir::createInstrumentPass(
    const std::string &ops, unsigned actions) {
  return std::make_unique<InstrumentPass>(ops, actions);
}
