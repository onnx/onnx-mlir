/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- ResourcePool.cpp --------------------------===//
//
// Garbage collects DenseResourceElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/Mlir/ResourcePool.hpp"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace onnx_mlir {

namespace {
const char *const namePrefix = "pool_";
}

size_t ResourcePool::ResourceHash::operator()(
    const DenseResourceElementsHandle &r) const {
  return hash_value(r);
}

/*static*/
ResourcePool &ResourcePool::create(MLIRContext *context) {
  return context->getLoadedDialect<BuiltinDialect>()
      ->addInterface<ResourcePool>(context);
}

/*static*/
ResourcePool *ResourcePool::get(MLIRContext *context) {
  return context->getLoadedDialect<BuiltinDialect>()
      ->getRegisteredInterface<ResourcePool>();
}

ResourcePool::ResourcePool(Dialect *dialect, MLIRContext *context)
    : Base(dialect), context(context), name(namePrefix) {}
ResourcePool::~ResourcePool() {}

DenseResourceElementsHandle ResourcePool::createResource(
    AsmResourceBlob &&blob) {
  name.resize(strlen(namePrefix));
  Twine(++nameCounter).toVector(name);
  DenseResourceElementsHandle r =
      DenseResourceElementsHandle::getManagerInterface(context).insert(
          name, std::move(blob));
  auto insertion = liveResources.insert(r);
  if (!insertion.second)
    llvm_unreachable("cannot insert existing resource");
  return r;
}

void ResourcePool::garbageCollect(const ResourceSet &newLiveResources) {
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

void ResourcePool::close() {
  garbageCollect({});
  active = false;
}

} // namespace onnx_mlir