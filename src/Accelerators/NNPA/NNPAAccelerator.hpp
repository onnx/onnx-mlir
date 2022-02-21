/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPAAccelerator.hpp
//-------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Accelerator class for NNPA
//
//===----------------------------------------------------------------------===//

#pragma once
#include "src/Accelerators/Accelerator.hpp"

namespace mlir {
class NNPAAccelerator : public Accelerator {
private:
  static bool initialized;

public:
  NNPAAccelerator();

  void prepareAccelerator(mlir::OwningModuleRef &module,
      mlir::MLIRContext &context, mlir::PassManager &pm,
      onnx_mlir::EmissionTargetType emissionTarget) override;
  bool isActive() override;
};

} // namespace mlir