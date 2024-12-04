/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ZHighDisposableGarbageCollector.hpp -----------------===//
//
// Garbage collects DisposableElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ZHIGH_GARBAGE_COLLECTOR_H
#define ONNX_MLIR_ZHIGH_GARBAGE_COLLECTOR_H

#include "mlir/Pass/PassInstrumentation.h"

namespace mlir {
class MLIRContext;
}

namespace onnx_mlir {
class DisposablePool;

namespace zhigh {

struct ZHighDisposableGarbageCollector : public mlir::PassInstrumentation {
  ZHighDisposableGarbageCollector(mlir::MLIRContext *context);
  ~ZHighDisposableGarbageCollector() override;

  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

private:
  DisposablePool &disposablePool;
};

} // namespace zhigh
} // namespace onnx_mlir
#endif
