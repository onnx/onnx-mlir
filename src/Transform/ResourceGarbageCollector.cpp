/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- ResourceGarbageCollector.cpp ---------------------===//
//
// Garbage collects DenseResourceElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ResourceGarbageCollector.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Transforms/Passes.h"

#include "src/Dialect/Mlir/ResourcePool.hpp"

using namespace mlir;

namespace onnx_mlir {

void ResourceGarbageCollector::runAfterPass(Pass *pass, Operation *op) {
  ResourcePool::ResourceSet liveResources;
  // TODO: traverse op and insert into liveResources the handle
  // of every DenseResourceElementsAttr
  resourcePool.garbageCollect(liveResources);
}

namespace {

struct ScrubResourcesPass
    : public PassWrapper<ScrubResourcesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScrubResourcesPass)

  ScrubResourcesPass(ResourcePool *resourcePool) : resourcePool(resourcePool) {}

  StringRef getArgument() const override { return "scrub-resources"; }

  void runOnOperation() final {
    // TODO: replace every DenseResourceElementsAttr with DenseElementsAttr
    getResourcePool()->close();
  }

  ResourcePool *getResourcePool() {
    return resourcePool ? resourcePool : ResourcePool::get(&getContext());
  }

  ResourcePool *resourcePool;
};

} // namespace

std::unique_ptr<mlir::Pass> createScrubResourcesPass(
    ResourcePool *resourcePool) {
  return std::make_unique<ScrubResourcesPass>(resourcePool);
}

} // namespace onnx_mlir