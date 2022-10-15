/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- ResourceGarbageCollector.hpp ---------------------===//
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

class ResourceGarbageCollector
    : public mlir::DialectInterface::Base<ResourceGarbageCollector> {
public:
  struct ResourceHash {
    size_t operator()(const mlir::DenseResourceElementsHandle &r) const;
  };
  using ResourceSet =
      std::unordered_set<mlir::DenseResourceElementsHandle, ResourceHash>;

  static ResourceGarbageCollector &create(mlir::MLIRContext *context);
  static ResourceGarbageCollector *get(mlir::MLIRContext *context);

  ResourceGarbageCollector(mlir::Dialect *dialect, mlir::MLIRContext *context);
  ~ResourceGarbageCollector();
  void insertResource(mlir::DenseResourceElementsHandle resource);
  void resetLiveResources(const ResourceSet &newLiveResources);
  bool isActive() const { return active; }
  void close();

private:
  mlir::MLIRContext *context;
  bool active = true;
  ResourceSet liveResources;
};

struct ResourceGCInstrumentation : public mlir::PassInstrumentation {
  ResourceGCInstrumentation(ResourceGarbageCollector &garbageCollector)
      : garbageCollector(garbageCollector) {}
  ~ResourceGCInstrumentation() override = default;

  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

private:
  ResourceGarbageCollector &garbageCollector;
};

} // namespace onnx_mlir
