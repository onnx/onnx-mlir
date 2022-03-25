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

extern bool InitAccelerators();

namespace onnx_mlir {
namespace accel {

class Accelerator {
public:
  Accelerator();
  virtual ~Accelerator();
  static std::vector<Accelerator *> getAcceleratorList();
  virtual bool isActive() const = 0;
  virtual void getOrLoadDialects(mlir::MLIRContext &context) const = 0;
  virtual void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::PassManager &pm,
      onnx_mlir::EmissionTargetType &emissionTarget) const = 0;
  virtual void registerDialects(mlir::DialectRegistry &registry) const = 0;
  virtual void initPasses(int optLevel) const = 0;

protected:
  // static llvm::SmallPtrSet<Accelerator *, 2> acceleratorTargets;
  static std::vector<Accelerator *> acceleratorTargets;
};

} // namespace accel
} // namespace onnx_mlir
