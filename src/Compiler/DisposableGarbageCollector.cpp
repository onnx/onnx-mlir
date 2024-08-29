/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ DisposableGarbageCollector.cpp --------------------===//
//
// Garbage collects DisposableElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#include "src/Compiler/DisposableGarbageCollector.hpp"

#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace onnx_mlir {

DisposableGarbageCollector::DisposableGarbageCollector(MLIRContext *context)
    : disposablePool(*DisposablePool::get<ONNXDialect>(context)) {}

DisposableGarbageCollector::~DisposableGarbageCollector() {}

void DisposableGarbageCollector::runAfterPass(Pass *pass, Operation *op) {
  if (!disposablePool.isActive())
    return;
  ModuleOp moduleOp = mlir::dyn_cast<ModuleOp>(op);
  if (!moduleOp)
    return;
  disposablePool.garbageCollectUnreachable(
      moduleOp, {{ONNXConstantOp::getOperationName(), "value"},
                    {ONNXConstantOfShapeOp::getOperationName(), "value"}});
}

} // namespace onnx_mlir
