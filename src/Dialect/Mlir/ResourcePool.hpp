/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- ResourcePool.hpp --------------------------===//
//
// Garbage collects DenseResourceElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectInterface.h"
#include "llvm/ADT/SmallString.h"

#include <unordered_set>

namespace onnx_mlir {

class ResourcePool : public mlir::DialectInterface::Base<ResourcePool> {
public:
  struct ResourceHash {
    size_t operator()(const mlir::DenseResourceElementsHandle &r) const;
  };
  using ResourceSet =
      std::unordered_set<mlir::DenseResourceElementsHandle, ResourceHash>;

  static ResourcePool &create(mlir::MLIRContext *context);

  static ResourcePool *get(mlir::MLIRContext *context);

  ResourcePool(mlir::Dialect *dialect, mlir::MLIRContext *context);

  ~ResourcePool();

  mlir::DenseResourceElementsHandle createResource(
      mlir::AsmResourceBlob &&blob);

  void garbageCollect(const ResourceSet &newLiveResources);

  bool isActive() const { return active; }

  void close();

private:
  // TODO: decide whether to remove the context field and instead pass context
  // arg to createResource()
  mlir::MLIRContext *context;
  bool active = true;
  ResourceSet liveResources;
  size_t nameCounter = 0;
  llvm::SmallString<32> name;
};

} // namespace onnx_mlir
