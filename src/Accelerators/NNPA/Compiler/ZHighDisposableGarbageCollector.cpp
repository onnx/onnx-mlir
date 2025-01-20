/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ZHighDisposableGarbageCollector.cpp -----------------===//
//
// Garbage collects DisposableElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Compiler/ZHighDisposableGarbageCollector.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace onnx_mlir {
namespace zhigh {

ZHighDisposableGarbageCollector::ZHighDisposableGarbageCollector(
    MLIRContext *context)
    : disposablePool(*DisposablePool::get<ONNXDialect>(context)) {}

ZHighDisposableGarbageCollector::~ZHighDisposableGarbageCollector() {}

void ZHighDisposableGarbageCollector::runAfterPass(Pass *pass, Operation *op) {
  if (!disposablePool.isActive())
    return;
  ModuleOp moduleOp = mlir::dyn_cast<ModuleOp>(op);
  if (!moduleOp)
    return;
  disposablePool.garbageCollectUnreachable(
      moduleOp, {{ONNXConstantOp::getOperationName(), "value"},
                    {ONNXConstantOfShapeOp::getOperationName(), "value"},
                    {ZHighStickifiedConstantOp::getOperationName(), "value"}});
}

} // namespace zhigh
} // namespace onnx_mlir
