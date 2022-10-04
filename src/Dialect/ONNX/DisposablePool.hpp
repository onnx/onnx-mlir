/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- DisposablePool.hpp ------------------------===//
//
// DisposablePool manages instances of DisposableElementsAttr.
// It creates them, maintains a record of them (in a "pool") until they are
// deemed unreachable, and it can be called to garbage collect those that are
// unreachable, and to "scrub" all occurrences in a module by replacing each
// with a DenseElementsAttr.
//
// Garbage collected and scrubbed DisposableElementsAttrs are removed from the
// pool and their reference to the underlying MemoryBuffer is cleared,
// decrementing the shared_ptr reference count.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectInterface.h"

#include <unordered_set>

namespace onnx_mlir {

class DisposablePool : public mlir::DialectInterface::Base<DisposablePool> {
  friend class ElementsAttrBuilder; // allow access to insert()
public:
  static DisposablePool &create(mlir::MLIRContext *context);

  static DisposablePool *get(mlir::MLIRContext *context);

  DisposablePool(mlir::Dialect *dialect, mlir::MLIRContext *context);
  ~DisposablePool();

  // // Create a DisposableElementsAttr and put it in the pool.
  // template <typename... Args>
  // mlir::DisposableElementsAttr createElementsAttr(Args &&...args) {
  //   auto d = mlir::DisposableElementsAttr::get(std::forward<Args>(args)...);
  //   insert(d);
  //   return d;
  // }

  // Disposes every DisposableElementsAttr in the pool which is unreachable
  // (doesn't appear in moduleOp).
  void garbageCollectUnreachable(mlir::ModuleOp moduleOp);

  // Disposes every DisposableElementsAttr and in moduleOp replaces each with a
  // DenseElementsAttr.
  void scrub(mlir::ModuleOp moduleOp);

  void close() {
    assert(pool.empty() && "pool must be scrubbed before close");
    active = false;
  }

  bool isActive() const { return active; }

private:
  using Pool = std::unordered_set<mlir::DisposableElementsAttributeStorage *>;

  void insert(mlir::DisposableElementsAttr d);
  void eraseUnreachable(const Pool &reachable);

  Pool pool;
  bool active = true;
};

} // namespace onnx_mlir
