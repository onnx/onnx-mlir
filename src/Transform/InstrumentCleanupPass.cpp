/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- InstrumentCleanupPass.cpp - Instrumentation -----------------===//
//
// Copyright 2025 The IBM Research Authors.
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

class InstrumentCleanupPass : public mlir::PassWrapper<InstrumentCleanupPass,
                                  OperationPass<func::FuncOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InstrumentCleanupPass)

  InstrumentCleanupPass() : {};
  InstrumentCleanupPass(const InstrumentCleanupPass &pass)
      : mlir::PassWrapper<InstrumentCleanupPass,
            OperationPass<func::FuncOp>>() {}

private:
public:
  StringRef getArgument() const override { return "instrument cleanup"; }

  StringRef getDescription() const override { return "instrument cleanup on ops."; }


  void runOnOperation() override {
    llvm::SmallVector<Operation *> eraseOpList;

    // Iterate on the operations nested in this function
    getOperation().walk([&](mlir::Operation *op) -> WalkResult {
      KrnlInstrumentOp firstOp = mlir::dyn_cast<KrnlInstrumentOp>(op);
      if (firstOp) {
        fprintf(stderr, "hi alex, has an instrument op\n");
        op->dump();
      }
      return WalkResult::advance();
    });
  }
};
} // namespace onnx_mlir

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createInstrumentCleanupPass() {
  return std::make_unique<InstrumentCleanupPass>();
}
