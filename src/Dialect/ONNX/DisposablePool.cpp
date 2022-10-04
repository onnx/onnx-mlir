/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- DisposablePool.cpp ------------------------===//
//
// DisposablePool manages instances of DisposableElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Dialect/ONNX/DisposableElementsAttributeStorage.hpp"

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

void DisposablePool::insert(DisposableElementsAttr d) {
  auto insertion = pool.insert(d.getImpl());
  if (!insertion.second)
    llvm_unreachable("cannot insert existing DisposableElementsAttr");
}

void DisposablePool::garbageCollectUnreachable(ModuleOp moduleOp) {
  Pool reachable;
  moduleOp.walk([&reachable, this](ONNXConstantOp constOp) {
    if (auto attr = constOp.value())
      if (auto elements = attr->dyn_cast<DisposableElementsAttr>()) {
        assert(this->pool.count(elements.getImpl()) == 1 &&
               "reachable disposables must be in the pool");
        reachable.insert(elements.getImpl());
      }
  });
  eraseUnreachable(reachable);
}

void DisposablePool::scrub(ModuleOp moduleOp) {
  moduleOp.walk([&](ONNXConstantOp constOp) {
    if (auto attr = constOp.value())
      if (auto elements = attr->dyn_cast<DisposableElementsAttr>()) {
        // TODO: Determine if we can encounter the same elements
        //       attribute twice and, if so, whether it's ok to rely on the
        //       storage uniquer to deduplicate or whether it's
        //       better to do it explicitly here somehow.
        assert(this->pool.count(elements.getImpl()) == 1 &&
               "reachable disposables must be in the pool");
        constOp.valueAttr(toDenseElementsAttr(elements));
      }
  });
  eraseUnreachable({});
}

void DisposablePool::eraseUnreachable(const Pool &reachable) {
  for (Pool::iterator it = pool.begin(); it != pool.end();) {
    DisposableElementsAttributeStorage *p = *it;
    if (pool.count(p) == 0) {
      // p is unreachable, so we reset the buffer payload shared_ptr
      // which decreases the reference count and, if it reached zero,
      // frees or closes the underlying MemoryBuffer's heap allocation or file.
      p->buffer.reset();
      p->reader = nullptr;
      it = pool.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace onnx_mlir