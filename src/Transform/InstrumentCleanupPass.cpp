/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- InstrumentCleanupPass.cpp - Instrumentation -----------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass that remove consecutive
// instrumentation operations (first with "before" tag and second with "after")
// as they do not measure anything.
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

class InstrumentCleanupPass : public mlir::PassWrapper<InstrumentCleanupPass,
                                  OperationPass<func::FuncOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InstrumentCleanupPass)

  InstrumentCleanupPass(){};
  InstrumentCleanupPass(const InstrumentCleanupPass &pass)
      : mlir::PassWrapper<InstrumentCleanupPass,
            OperationPass<func::FuncOp>>() {}

private:
public:
  StringRef getArgument() const override { return "instrument-cleanup"; }

  StringRef getDescription() const override {
    return "instrument cleanup on ops.";
  }

  void runOnOperation() override {
    llvm::SmallVector<Operation *> eraseOpList;
    bool skipNext = false;

    // Iterate on the operations nested in this function
    getOperation().walk([&](mlir::Operation *op) -> WalkResult {
      if (skipNext) {
        skipNext = false;
        return WalkResult::advance();
      }
      KrnlInstrumentOp firstInstrOp = mlir::dyn_cast<KrnlInstrumentOp>(op);
      // Check if we have a first instrumentation op with instr before.
      if (!firstInstrOp)
        return WalkResult::advance();
      uint64_t firstTag = firstInstrOp.getTag();
      // skip if not before, or if this call initializes the instrumentation.
      if (!IS_INSTRUMENT_BEFORE_OP(firstTag) || IS_INSTRUMENT_INIT(firstTag))
        return WalkResult::advance();
      // Check if we have a second instrumentation op with instr after.
      Operation *nextOp = op->getNextNode();
      if (!nextOp)
        return WalkResult::advance();
      KrnlInstrumentOp secondInstrOp = mlir::dyn_cast<KrnlInstrumentOp>(nextOp);
      if (!secondInstrOp)
        return WalkResult::advance();
      uint64_t secondTag = secondInstrOp.getTag();
      // skip if not after, or if this call initializes the instrumentation.
      if (!IS_INSTRUMENT_AFTER_OP(secondTag) || IS_INSTRUMENT_INIT(secondTag))
        return WalkResult::advance();
      // Could check opName but we already have a before/after pair, it can only
      // be of the same op.
      // Schedule both instrumentation to be removed as there is nothing between
      // the start and the stop of the instrumentation.
      eraseOpList.emplace_back(op);
      eraseOpList.emplace_back(nextOp);
      skipNext = true;
      return WalkResult::advance();
    });
    // Remove ops.
    for (Operation *op : eraseOpList)
      op->erase();
  }
};
} // namespace onnx_mlir

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createInstrumentCleanupPass() {
  return std::make_unique<InstrumentCleanupPass>();
}
