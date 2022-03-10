/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPAAccelerator.hpp ----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// ===========================================================================
//
// Accelerator class for NNPA
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
  void prepareAccelerator(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::MLIRContext &context, mlir::PassManager &pm,
      onnx_mlir::EmissionTargetType emissionTarget) const final;
};

} // namespace nnpa
} // namespace accel
} // namespace onnx_mlir
