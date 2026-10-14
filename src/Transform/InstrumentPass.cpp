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
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Compiler/OptionUtils.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_INSTRUMENTPASS
#include "src/Transform/Passes.h.inc"
} // namespace onnx_mlir

namespace {
/*!
 * This pass insert KrnlInstrumentOp before and after each ops
 */

class InstrumentPass
    : public onnx_mlir::impl::InstrumentPassBase<InstrumentPass> {

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
      : onnx_mlir::impl::InstrumentPassBase<InstrumentPass>(),
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

  // Base tag: time/memory report flags only, no before/after position bits.
  uint64_t reportFlagsTag() const {
    int64_t tag;
    INIT_INSTRUMENT(tag);
    if (reportTime)
      SET_INSTRUMENT_REPORT_TIME(tag);
    if (reportMemory)
      SET_INSTRUMENT_REPORT_MEMORY(tag);
    return tag;
  }

  uint64_t initTag() const { return reportFlagsTag(); }

  uint64_t beforeTag() const {
    int64_t tag = reportFlagsTag();
    SET_INSTRUMENT_BEFORE_OP(tag);
    return tag;
  }

  uint64_t afterTag() const {
    int64_t tag = reportFlagsTag();
    SET_INSTRUMENT_AFTER_OP(tag);
    return tag;
  }

  void runOnOperation() override {
    assert(instrumentOps != "" && instrumentOps != "NONE" &&
           "should only be here if we have something to instrument");

    allowedOps.setRegexString(instrumentOps);
    bool hasInitializedRuntime = false;

    // Pre-order walk so we can skip ONNXFusedOp bodies with WalkResult::skip().
    getOperation().walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::Operation *op) -> WalkResult {
          // Do not profile operations that return a None value (e.g.
          // onnx.NoValue). Somehow such none-returned operations cause messy
          // output, For example, with --profile-ir=ZHigh for the mnist-12
          // model, it mixed version error with profiling info.
          // ```
          // #  0) before zlow.stickModel is running on hardware that is not
          // compatible with the zDNN library that this model was compiled for
          // (version num %llu.%llu.%llu). Please check that the model is
          // running on hardware with an integrated accelerator for AI (z16 +)
          // that supports the required zDNN library version.
          // ```
          Location loc = op->getLoc();
          OpBuilder opBuilder(op);

          if (op->getNumResults() == 1 &&
              isa<NoneType>(op->getResult(0).getType()))
            return WalkResult::advance();
          // Skip other instrument ops.
          if (isa<KrnlInstrumentOp>(op) || isa<KrnlInstrumentInitOp>(op) ||
              isa<ONNXPrintSignatureOp>(op))
            return WalkResult::advance();

          // ONNXFusedOp uses "onnx.fused.<kind>" as its profiling name and must
          // not recurse into the body (body ops are an internal lowering
          // detail). All other ops use their dialect name.  Both cases share
          // the same emit logic; only the name and the post-instrument walk
          // result differ.
          bool isFused = isa<ONNXFusedOp>(op);
          std::string instrName = onnx_mlir::getProfilingName(op);

          if (allowedOps.isEnabled(instrName)) {
            std::string nodeName = onnx_mlir::getNodeNameInPresenceOfOpt(op);
            auto emitInstrument = [&](OpBuilder &b, uint64_t tag) {
              if (!hasInitializedRuntime) {
                mlir::KrnlInstrumentInitOp::create(b, loc, initTag());
                hasInitializedRuntime = true;
              }
              mlir::KrnlInstrumentOp::create(b, loc, instrName, nodeName, tag);
            };
            if (instrumentBefore)
              emitInstrument(opBuilder, beforeTag());
            // Can not insert after Op (e.g. ONNXYieldOp) with IsTerminator
            // trait.
            if (instrumentAfter && !op->hasTrait<OpTrait::IsTerminator>()) {
              opBuilder.setInsertionPointAfter(op);
              emitInstrument(opBuilder, afterTag());
            }
          }
          // Skip FusedOp bodies; advance normally for all other ops.
          return isFused ? WalkResult::skip() : WalkResult::advance();
        });
  }
};

} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createInstrumentPass(
    const std::string &ops, unsigned actions) {
  return std::make_unique<InstrumentPass>(ops, actions);
}
} // namespace onnx_mlir
