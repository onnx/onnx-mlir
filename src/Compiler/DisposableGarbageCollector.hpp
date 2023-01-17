/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ DisposableGarbageCollector.hpp --------------------===//
//
// Garbage collects DisposableElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/PassInstrumentation.h"

namespace mlir {
class MLIRContext;
}

namespace onnx_mlir {

class DisposablePool;

struct DisposableGarbageCollector : public mlir::PassInstrumentation {
  DisposableGarbageCollector(mlir::MLIRContext *context);
  ~DisposableGarbageCollector() override;

  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

private:
  DisposablePool &disposablePool;
};

} // namespace onnx_mlir
