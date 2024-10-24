/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- DisposablePool.cpp ------------------------===//
//
// DisposablePool manages instances of DisposableElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"

#include "mlir/IR/Threading.h"
#include "llvm/ADT/DenseMap.h"

#include <atomic>

using namespace mlir;

namespace onnx_mlir {

DisposablePool::DisposablePool(Dialect *dialect, MLIRContext *context)
    : Base(dialect), pool(), mutex() {}
DisposablePool::~DisposablePool() {}

ElementsAttr DisposablePool::createElementsAttr(ShapedType type,
    BType bufferBType, ArrayRef<int64_t> strides,
    const DisposableElementsAttr::Buffer &buffer,
    DisposableElementsAttr::Transformer transformer) {
  static std::atomic<size_t> counter{0};
  size_t id = ++counter;
  auto disposable = DisposableElementsAttr::create(
      type, id, bufferBType, strides, buffer, std::move(transformer));
  if (insert(disposable)) {
    return disposable;
  } else {
    auto dense = disposable.toDenseElementsAttr();
    disposable.dispose();
    return dense;
  }
}

namespace {
template <typename Action>
void walkOpsAttrs(ModuleOp moduleOp, DisposablePool::OpAttrDictionary opsAttrs,
    const Action &act) {
  llvm::SmallDenseMap<StringRef, StringRef> opAttrMap(
      opsAttrs.begin(), opsAttrs.end());
  moduleOp.walk([&opAttrMap, &act](Operation *op) {
    auto opAttr = opAttrMap.find(op->getName().getIdentifier());
    if (opAttr != opAttrMap.end()) {
      StringRef attrName = opAttr->second;
      if (auto attr = op->getAttrOfType<DisposableElementsAttr>(attrName))
        act(op, attrName, attr);
    }
  });
}
} // namespace

void DisposablePool::garbageCollectUnreachable(
    ModuleOp moduleOp, OpAttrDictionary opsAttrs) {
  Pool reachable;
  walkOpsAttrs(moduleOp, opsAttrs,
      [&reachable](Operation *op, StringRef attrName,
          DisposableElementsAttr disposable) {
        reachable.try_emplace(disposable.getId(), disposable);
      });

  {
    const std::lock_guard<std::mutex> lock(mutex);

    for (auto &entry : reachable)
      assert(pool.count(entry.first) == 1 &&
             "reachable disposables must be in the pool");
    eraseUnreachable(reachable);
  }
}

void DisposablePool::scrub(ModuleOp moduleOp, OpAttrDictionary opsAttrs) {
  using Translation = std::pair<DisposableElementsAttr, DenseElementsAttr>;
  std::unordered_map<size_t, Translation> translations;
  walkOpsAttrs(moduleOp, opsAttrs,
      [&translations](Operation *op, StringRef attrName,
          DisposableElementsAttr disposable) {
        translations.try_emplace(
            disposable.getId(), std::make_pair(disposable, nullptr));
      });

  {
    // The mutex protects access to the iterator 'next' which is progressed as
    // translations are fetched in batches to be processed by the parallel
    // workers in work().
    std::mutex translationMutex;
    auto next = translations.begin();
    // Returns a range of consecutive translations to work on.
    // Returns an empty range when there are no more translations left.
    auto fetchBatch = [&translationMutex, &translations, &next]() {
      const std::lock_guard<std::mutex> lock(translationMutex);
      auto batchBegin = next;
      auto batchEnd = batchBegin;
      size_t count = 0, aggregateSize = 0;
      // To avoid excessive parallel coordination overhead if there are many
      // small attributes: keep growing the batch until next has at least 10
      // attributes or their aggregate size (elements count) is at least 1000.
      constexpr size_t minCount = 10, minAggregateCount = 1000;
      while (count < minCount && aggregateSize < minAggregateCount &&
             batchEnd != translations.end()) {
        auto [disposable, _] = next->second;
        aggregateSize += disposable.size();
        ++count;
        ++batchEnd;
      }
      next = batchEnd;
      return llvm::make_range(batchBegin, batchEnd);
    };
    // Parallel worker body: Fetch and process batches until there are no more.
    auto work = [&fetchBatch, &translationMutex](size_t threadNumber) {
      for (;;) {
        auto batch = fetchBatch();
        if (batch.empty())
          break;
        for (auto &[id, translation] : batch) {
          auto &[disposable, dense] = translation;
          dense = disposable.toDenseElementsAttr();
          {
            const std::lock_guard<std::mutex> lock(translationMutex);
            disposable.dispose();
          }
        }
      }
    };
    MLIRContext *ctx = moduleOp.getContext();
    parallelFor(ctx, 0, ctx->getNumThreads(), work);
  }

  walkOpsAttrs(moduleOp, opsAttrs,
      [&translations](Operation *op, StringRef attrName,
          DisposableElementsAttr disposable) {
        DenseElementsAttr dense = translations.at(disposable.getId()).second;
        op->setAttr(attrName, dense);
      });

  {
    const std::lock_guard<std::mutex> lock(mutex);

    for (const auto &t : translations)
      assert(pool.count(t.first) == 1 &&
             "scrubbed disposables must be in the pool");
    eraseUnreachable({});
  }
}

void DisposablePool::close() {
  const std::lock_guard<std::mutex> lock(mutex);

  assert(pool.empty() && "pool must be scrubbed before close");
  active = false;
}

bool DisposablePool::isActive() const {
  const std::lock_guard<std::mutex> lock(mutex);

  return active;
}

bool DisposablePool::insert(DisposableElementsAttr disposable) {
  const std::lock_guard<std::mutex> lock(mutex);

  if (!active)
    return false;

  auto insertion = pool.try_emplace(disposable.getId(), disposable);
  if (!insertion.second)
    llvm_unreachable("cannot insert existing DisposableElementsAttr");
  return true;
}

void DisposablePool::eraseUnreachable(const Pool &reachable) {
  // Assumes caller holds the mutex.
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