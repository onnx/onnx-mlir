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

#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ScrubDisposablePass
    : public PassWrapper<ScrubDisposablePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScrubDisposablePass)

  ScrubDisposablePass(DisposablePool *disposablePool)
      : disposablePool(disposablePool) {}

  StringRef getArgument() const override { return "scrub-disposable"; }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    getDisposablePool()->scrub(moduleOp);
  }

  DisposablePool *getDisposablePool() {
    // The disposablePool may be unknown (nullptr) at the time of construction
    // of the pass, so fall back to looking it up when the pass runs.
    return disposablePool ? disposablePool : DisposablePool::get(&getContext());
  }

  DisposablePool *disposablePool;
};

} // namespace

std::unique_ptr<mlir::Pass> createScrubDisposablePass(
    DisposablePool *disposablePool) {
  return std::make_unique<ScrubDisposablePass>(disposablePool);
}

} // namespace onnx_mlir