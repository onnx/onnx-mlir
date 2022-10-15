/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- ResourceGarbageCollector.cpp ---------------------===//
//
// Garbage collects DenseResourceElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/ResourceGarbageCollector.hpp"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace onnx_mlir {

size_t ResourceGarbageCollector::ResourceHash::operator()(
    const DenseResourceElementsHandle &r) const {
  return hash_value(r);
}

ResourceGarbageCollector::ResourceGarbageCollector(Dialect *dialect) : Base(dialect) {}
ResourceGarbageCollector::~ResourceGarbageCollector() {}

void ResourceGarbageCollector::insertResource(
    DenseResourceElementsHandle resource) {
  auto insertion = liveResources.insert(resource);
  if (!insertion.second)
    llvm_unreachable("cannot insert existing resource");
}

void ResourceGarbageCollector::resetLiveResources(
    const ResourceSet &newLiveResources) {
  assert(llvm::all_of(newLiveResources, [this](const auto &r) {
    return this->liveResources.count(r) == 1;
  }) && "new live resources must be included in the old ones");

  for (ResourceSet::iterator it = liveResources.begin();
       it != liveResources.end();) {
    if (newLiveResources.count(*it) == 0) {
      DenseResourceElementsHandle r = *it;
      r.getResource()->setBlob({}); // Free blob.
      it = liveResources.erase(it);
    } else {
      ++it;
    }
  }
}

void ResourceGarbageCollector::close() {
  resetLiveResources({});
  closed = true;
}

void ResourceGCInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  ResourceGarbageCollector::ResourceSet liveResources;
  // TODO: traverse op and insert into liveResources the handle
  // of every DenseResourceElementsAttr
  garbageCollector.resetLiveResources(liveResources);
}

namespace {

struct ScrubResourcesPass
    : public PassWrapper<ScrubResourcesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScrubResourcesPass)

  ScrubResourcesPass(ResourceGarbageCollector &resourceGarbageCollector)
      : resourceGarbageCollector(resourceGarbageCollector) {}

  StringRef getArgument() const override { return "resource-gc"; }

  void runOnOperation() final {
    // TODO: replace every DenseResourceElementsAttr with DenseElementsAttr
    resourceGarbageCollector.close();
  }

  ResourceGarbageCollector &resourceGarbageCollector;
};

} // namespace

std::unique_ptr<mlir::Pass> createScrubResourcesPass(
    ResourceGarbageCollector &resourceGarbageCollector) {
  return std::make_unique<ScrubResourcesPass>(resourceGarbageCollector);
}

} // namespace onnx_mlir