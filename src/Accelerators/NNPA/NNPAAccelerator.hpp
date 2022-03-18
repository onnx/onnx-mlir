/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPAAccelerator.hpp ----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// ===========================================================================
//
// Accelerator class for IBM Telum coprocessor
//
//===---------------------------------------------------------------------===//

#pragma once

#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir {
namespace accel {
namespace nnpa {

/// Singleton class.
class NNPAAccelerator final : public Accelerator {
public:
  ~NNPAAccelerator() {}

  /// Retrieve the singleton object (non-thread safe).
  static Accelerator *getInstance();

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const Accelerator *accel) {
    return accel->getKind() == Accelerator::Kind::NNPA;
  }
  static bool classof(const NNPAAccelerator *) { return true; }

private:
  static NNPAAccelerator singleton;

  NNPAAccelerator() : Accelerator(Accelerator::Kind::NNPA) {}

  // Prevent copy construction and assignement.
  NNPAAccelerator(NNPAAccelerator &other) = delete;
  void operator=(const NNPAAccelerator &) = delete;

  /// Initialize the accelerator.
  void prepare(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::MLIRContext &context, mlir::PassManager &pm,
      onnx_mlir::EmissionTargetType emissionTarget) const final;
};

} // namespace nnpa
} // namespace accel
} // namespace onnx_mlir
