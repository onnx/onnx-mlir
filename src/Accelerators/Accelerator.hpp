/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Accelerator.hpp ---------------------------===//
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
  /// Kinds of accelerators.
  enum class Kind {
#include "src/Accelerators/AcceleratorKinds.hpp"
  };

  Accelerator(Kind kind) : kind(kind) {}
  virtual ~Accelerator();

  /// Getter for the kind of this accelerator.
  Kind getKind() const { return kind; }

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

  /// Kind of accelerator.
  Kind kind;
};

} // namespace accel
} // namespace onnx_mlir
