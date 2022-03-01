/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Accelerator.hpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// OMAccelerator base class
//
//===----------------------------------------------------------------------===//
#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#include <vector>

namespace mlir {
class Accelerator {
public:
  Accelerator();
  static std::vector<Accelerator *> *getAcceleratorList();
  virtual bool isActive() = 0;
  virtual void prepareAccelerator(mlir::OwningOpRef<ModuleOp> &module,
      mlir::MLIRContext &context, mlir::PassManager &pm,
      onnx_mlir::EmissionTargetType emissionTarget) = 0;

private:
  static std::vector<Accelerator *> *acceleratorTargets;
};
} // namespace mlir