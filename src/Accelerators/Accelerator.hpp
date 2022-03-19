/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Accelerator.hpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Accelerator base class
//
//===----------------------------------------------------------------------===//

#pragma once

#include "include/onnx-mlir/Compiler/OMCompilerTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include <vector>

namespace onnx_mlir {
namespace accel {

class Accelerator {
public:
  Accelerator();
  virtual ~Accelerator();
  static std::vector<Accelerator *> getAcceleratorList();
  virtual bool isActive() const = 0;
  virtual void prepareAccelerator(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::MLIRContext &context, mlir::PassManager &pm,
      onnx_mlir::EmissionTargetType emissionTarget) const = 0;

protected:
  // static llvm::SmallPtrSet<Accelerator *, 2> acceleratorTargets;
  static std::vector<Accelerator *> acceleratorTargets;
};

} // namespace accel
} // namespace onnx_mlir
