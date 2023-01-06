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

#include <unordered_map>

namespace onnx_mlir {

class DisposablePool : public mlir::DialectInterface::Base<DisposablePool> {
public:
  template <typename Dialect>
  static DisposablePool &create(mlir::MLIRContext *context) {
    return context->getLoadedDialect<Dialect>()
        ->template addInterface<DisposablePool>(context);
  }

  template <typename Dialect>
  static DisposablePool *get(mlir::MLIRContext *context) {
    return context->getLoadedDialect<Dialect>()
        ->template getRegisteredInterface<DisposablePool>();
  }

  using OpAttrPair = std::pair<llvm::StringRef, llvm::StringRef>;
  using OpAttrDictionary = llvm::ArrayRef<OpAttrPair>;

  // Disposes every DisposableElementsAttr and in moduleOp replaces each with a
  // DenseElementsAttr. This is irreversible and is called when we
  // are done transforming the ONNX dialect, just before we lower it.
  static void scrub(mlir::ModuleOp moduleOp, OpAttrDictionary opsAttrs,
      DisposablePool *disposablepool);

  DisposablePool(mlir::Dialect *dialect, mlir::MLIRContext *context);
  ~DisposablePool();

  // Create a ElementsAttr and put it in the DisposablePool if it's active,
  // otherwise returns conversion to DenseElementsAttr.
  mlir::ElementsAttr createElementsAttr(mlir::ShapedType type,
      BType bufferBType, llvm::ArrayRef<int64_t> strides,
      const mlir::DisposableElementsAttr::Buffer &buffer,
      mlir::DisposableElementsAttr::Transformer transformer);

  // Disposes every DisposableElementsAttr in the pool which is unreachable
  // (doesn't appear in moduleOp).
  void garbageCollectUnreachable(
      mlir::ModuleOp moduleOp, OpAttrDictionary opsAttrs);

  // Can be called when the pool is empty, namely after calling scrub(), to
  // ensure that no more DisposableElementsAttr instances are created, i.e.
  // will cause all future calls to insert() to fail.
  void close() {
    assert(pool.empty() && "pool must be scrubbed before close");
    active = false;
  }

  bool isActive() const { return active; }

private:
  // TODO: Change to unordered_set with C++20 where we can key set members by
  //       id and find set members by id without constructing a member object.
  using Pool = std::unordered_map<size_t, mlir::DisposableElementsAttr>;
  using Scrubbed = std::unordered_map<size_t, mlir::DenseElementsAttr>;

  // Record all instances of DisposableElementsAttr as they are created.
  // Returns true on success and false if the pool is not active.
  bool insert(mlir::DisposableElementsAttr disposable);

  static Scrubbed doScrub(mlir::ModuleOp moduleOp, OpAttrDictionary opsAttrs);

  void flushAfterScrub(const Scrubbed &scrubbed);

  void eraseUnreachable(const Pool &reachable);

  Pool pool;
  bool active = true;
};

} // namespace onnx_mlir
