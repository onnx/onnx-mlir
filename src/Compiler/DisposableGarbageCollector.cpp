/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ DisposableGarbageCollector.cpp --------------------===//
//
// Garbage collects DisposableElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#include "src/Compiler/DisposableGarbageCollector.hpp"

#include "src/Dialect/ONNX/DisposablePool.hpp"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace onnx_mlir {

void DisposableGarbageCollector::runAfterPass(Pass *pass, Operation *op) {
  if (!disposablePool.isActive())
    return;
  ModuleOp moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp)
    return;
  disposablePool.garbageCollectUnreachable(moduleOp);
}

} // namespace onnx_mlir