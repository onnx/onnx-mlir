/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- ResourceGarbageCollector.hpp ---------------------===//
//
// Garbage collects DenseResourceElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Pass/PassInstrumentation.h"

#include <unordered_set>

namespace onnx_mlir {

class ResourcePool;

struct ResourceGarbageCollector : public mlir::PassInstrumentation {
  ResourceGarbageCollector(ResourcePool &resourcePool)
      : resourcePool(resourcePool) {}
  ~ResourceGarbageCollector() override = default;

  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

private:
  ResourcePool &resourcePool;
};

} // namespace onnx_mlir
