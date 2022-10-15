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
#include "mlir/Pass/PassInstrumentation.h"

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
  void insertResource(mlir::DenseResourceElementsHandle resource);
  void garbageCollect(const ResourceSet &newLiveResources);
  bool isActive() const { return active; }
  void close();

private:
  mlir::MLIRContext *context; // TODO: decide whether to remove this
  bool active = true;
  ResourceSet liveResources;
};

} // namespace onnx_mlir
