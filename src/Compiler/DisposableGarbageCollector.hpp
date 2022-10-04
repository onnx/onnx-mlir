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

namespace onnx_mlir {

class DisposablePool;

struct DisposableGarbageCollector : public mlir::PassInstrumentation {
  DisposableGarbageCollector(DisposablePool &disposablePool)
      : disposablePool(disposablePool) {}
  ~DisposableGarbageCollector() override = default;

  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

private:
  DisposablePool &disposablePool;
};

} // namespace onnx_mlir
