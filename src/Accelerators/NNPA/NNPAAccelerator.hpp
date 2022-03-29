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
namespace nnpa {

class NNPAAccelerator final : public Accelerator {
private:
  static bool initialized;

public:
  NNPAAccelerator();

  bool isActive() const final;
  virtual void getOrLoadDialects(mlir::MLIRContext &context) const final;
  virtual void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::PassManager &pm,
      onnx_mlir::EmissionTargetType &emissionTarget) const final;
  virtual void registerDialects(mlir::DialectRegistry &registry) const final;
  virtual void initPasses(int optLevel) const final;
};

} // namespace nnpa
} // namespace accel
} // namespace onnx_mlir
