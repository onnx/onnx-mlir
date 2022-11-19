/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ScrubDisposablePass.cpp ------------------------===//
//
// Replaces each DisposableElementsAttr with a DenseElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

#include "src/Dialect/ONNX/DisposablePool.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ScrubDisposablePass
    : public PassWrapper<ScrubDisposablePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScrubDisposablePass)

  ScrubDisposablePass(DisposablePool *disposablePool, bool closeAfter)
      : disposablePool(disposablePool), closeAfter(closeAfter) {}

  StringRef getArgument() const override { return "scrub-disposable"; }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    DisposablePool *pool = getDisposablePool();
    DisposablePool::scrub(moduleOp, pool);
    if (closeAfter && pool)
      pool->close();
  }

  DisposablePool *getDisposablePool() {
    // The disposablePool may be unknown (nullptr) at the time of construction
    // of the pass, so fall back to looking it up when the pass runs.
    return disposablePool ? disposablePool : DisposablePool::get(&getContext());
  }

  DisposablePool * const disposablePool;
  const bool closeAfter;
};

} // namespace

std::unique_ptr<mlir::Pass> createScrubDisposablePass(
    DisposablePool *disposablePool, bool closeAfter) {
  return std::make_unique<ScrubDisposablePass>(disposablePool, closeAfter);
}

} // namespace onnx_mlir