/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- ZHighScrubDisposablePass.cpp --------------------===//
//
// Replaces each DisposableElementsAttr with a DenseElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace zhigh {
namespace {

struct ZHighScrubDisposablePass
    : public PassWrapper<ZHighScrubDisposablePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHighScrubDisposablePass)

  ZHighScrubDisposablePass(bool closeAfter) : closeAfter(closeAfter) {}

  StringRef getArgument() const override { return "zhigh-scrub-disposable"; }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    DisposablePool *pool = getDisposablePool();
    pool->scrub(moduleOp,
        {{ONNXConstantOp::getOperationName(), "value"},
            {ONNXConstantOfShapeOp::getOperationName(), "value"},
            {ZHighStickifiedConstantOp::getOperationName(), "value"}});
    if (closeAfter)
      pool->close();
  }

  DisposablePool *getDisposablePool() {
    // It can be hard to get the MLIRContext at the time of construction
    // of the pass, so we look it up the first time the pass is run.
    if (!disposablePool)
      disposablePool = DisposablePool::get<ONNXDialect>(&getContext());
    return disposablePool;
  }

  const bool closeAfter;
  DisposablePool *disposablePool = nullptr;
};

} // namespace

std::unique_ptr<mlir::Pass> createZHighScrubDisposablePass(bool closeAfter) {
  return std::make_unique<ZHighScrubDisposablePass>(closeAfter);
}

} // namespace zhigh
} // namespace onnx_mlir
