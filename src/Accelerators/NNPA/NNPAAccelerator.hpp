/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPAAccelerator.hpp ----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// ===========================================================================
//
// Accelerator support for the IBM Telum coprocessor.
//
//===---------------------------------------------------------------------===//

#pragma once

#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir {
namespace accel {

/// Singleton class to construct an NNPA accelerator.
class NNPAAccelerator final : public Accelerator {
private:
  static NNPAAccelerator *instance;
  NNPAAccelerator();

public:
  /// Singleton should not be clonable or assignable.
  NNPAAccelerator(NNPAAccelerator &) = delete;
  void operator=(const NNPAAccelerator &) = delete;

  ~NNPAAccelerator();

  /// Creates an instance on the first invocation. Subsequernt invocations
  /// return the existing instance.
  static NNPAAccelerator *getInstance();

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const Accelerator *accel) {
    return accel->getKind() == Accelerator::Kind::NNPA;
  }
  static bool classof(const NNPAAccelerator *) { return true; }

  bool isActive() const final;
  virtual void getOrLoadDialects(mlir::MLIRContext &context) const final;
  virtual void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::PassManager &pm,
      onnx_mlir::EmissionTargetType &emissionTarget) const final;
  virtual void registerDialects(mlir::DialectRegistry &registry) const final;
  virtual void initPasses(int optLevel) const final;
};

} // namespace accel
} // namespace onnx_mlir
