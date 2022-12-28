/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- DisposablePool.cpp ------------------------===//
//
// DisposablePool manages instances of DisposableElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DisposablePool.hpp"

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include <atomic>

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

DisposableElementsAttr DisposablePool::createDisposableElementsAttr(
    ShapedType type, BType bufferBType, ArrayRef<int64_t> strides,
    const mlir::DisposableElementsAttr::Buffer &buffer,
    DisposableElementsAttr::Transformer transformer) {
  static std::atomic<size_t> counter{0};
  size_t id = ++counter;
  auto d = DisposableElementsAttr::create(
      type, id, bufferBType, strides, buffer, std::move(transformer));
  insert(d);
  return d;
}

void DisposablePool::insert(DisposableElementsAttr disposable) {
  // TODO: make this thread safe
  assert(isActive());
  auto insertion = pool.try_emplace(disposable.getId(), disposable);
  if (!insertion.second)
    llvm_unreachable("cannot insert existing DisposableElementsAttr");
}

/*static*/
template <typename CONST_OP>
void DisposablePool::collectReachable(ModuleOp moduleOp, Pool &reachable) {
  moduleOp.walk([&reachable](CONST_OP constOp) {
    if (auto attr = constOp.value())
      if (auto disposable = attr->template dyn_cast<DisposableElementsAttr>()) {
        reachable.try_emplace(disposable.getId(), disposable);
      }
  });
}

void DisposablePool::garbageCollectUnreachable(ModuleOp moduleOp) {
  Pool reachable;
  collectReachable<ONNXConstantOp>(moduleOp, reachable);
  collectReachable<ONNXConstantOfShapeOp>(moduleOp, reachable);
  for (auto &entry : reachable)
    assert(pool.count(entry.first) == 1 &&
           "reachable disposables must be in the pool");
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
template <typename CONST_OP>
void DisposablePool::scrubConstants(ModuleOp moduleOp, Scrubbed &scrubbed) {
  moduleOp.walk([&scrubbed](CONST_OP constOp) {
    if (auto attr = constOp.value())
      if (auto disposable = attr->template dyn_cast<DisposableElementsAttr>()) {
        auto insertion = scrubbed.try_emplace(disposable.getId(), nullptr);
        auto iter = insertion.first;
        if (insertion.second) { // disposable was inserted
          iter->second = disposable.toDenseElementsAttr();
        }
        constOp.valueAttr(iter->second);
      }
  });
}

/*static*/
auto DisposablePool::doScrub(ModuleOp moduleOp) -> Scrubbed {
  Scrubbed scrubbed;
  scrubConstants<ONNXConstantOp>(moduleOp, scrubbed);
  scrubConstants<ONNXConstantOfShapeOp>(moduleOp, scrubbed);
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