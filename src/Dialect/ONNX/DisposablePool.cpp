/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- DisposablePool.cpp ------------------------===//
//
// DisposablePool manages instances of DisposableElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DisposablePool.hpp"

#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

/*static*/
DisposablePool &DisposablePool::create(MLIRContext *context) {
  return context->getLoadedDialect<ONNXDialect>()->addInterface<DisposablePool>(
      context);
}

/*static*/
DisposablePool *DisposablePool::get(MLIRContext *context) {
  return context->getLoadedDialect<ONNXDialect>()
      ->getRegisteredInterface<DisposablePool>();
}

DisposablePool::DisposablePool(Dialect *dialect, MLIRContext *context)
    : Base(dialect), pool() {}
DisposablePool::~DisposablePool() {}

void DisposablePool::insert(DisposableElementsAttr disposable) {
  // TODO: make this thread safe
  auto insertion = pool.try_emplace(disposable.getId(), disposable);
  if (!insertion.second)
    llvm_unreachable("cannot insert existing DisposableElementsAttr");
}

DisposableElementsAttr DisposablePool::lookup(size_t id) const {
  auto found = pool.find(id);
  if (found == pool.end())
    return nullptr;
  return found->second;
}

void DisposablePool::garbageCollectUnreachable(ModuleOp moduleOp) {
  Pool reachable;
  moduleOp.walk([&reachable, this](ONNXConstantOp constOp) {
    if (auto attr = constOp.value())
      if (auto disposable = attr->dyn_cast<DisposableElementsAttr>()) {
        assert(this->pool.count(disposable.getId()) == 1 &&
               "reachable disposables must be in the pool");
        reachable.try_emplace(disposable.getId(), disposable);
      }
  });
  eraseUnreachable(reachable);
}

/*static*/
void DisposablePool::scrub(
    mlir::ModuleOp moduleOp, DisposablePool *disposablePool) {
  Scrubbed scrubbed = doScrub(moduleOp);
  if (disposablePool)
    disposablePool->flushAfterScrub(scrubbed);
}

/*static*/
auto DisposablePool::doScrub(ModuleOp moduleOp) -> Scrubbed {
  Scrubbed scrubbed;
  moduleOp.walk([&scrubbed](ONNXConstantOp constOp) {
    if (auto attr = constOp.value())
      if (auto disposable = attr->dyn_cast<DisposableElementsAttr>()) {
        auto insertion = scrubbed.try_emplace(disposable.getId(), nullptr);
        auto iter = insertion.first;
        if (insertion.second) { // disposable was inserted
          iter->second = toDenseElementsAttr(disposable);
        }
        constOp.valueAttr(iter->second);
      }
  });
  return scrubbed;
}

void DisposablePool::flushAfterScrub(const Scrubbed &scrubbed) {
  for (const auto &s : scrubbed)
    assert(
        pool.count(s.first) == 1 && "scrubbed disposables must be in the pool");
  eraseUnreachable({});
}

void DisposablePool::eraseUnreachable(const Pool &reachable) {
  for (Pool::iterator it = pool.begin(); it != pool.end();) {
    if (reachable.count(it->first) == 0) {
      // The attribute is unreachable, so we reset the buffer payload shared_ptr
      // which decreases the reference count and, if it reached zero,
      // frees or closes the underlying MemoryBuffer's heap allocation or file.
      it->second.dispose();
      it = pool.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace onnx_mlir