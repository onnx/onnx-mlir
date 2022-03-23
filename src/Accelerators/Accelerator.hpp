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

namespace onnx_mlir {
namespace accel {

class Accelerator {
public:
  /// Kinds of accelerators.
  enum class Kind {
    NNPA, // IBM Telum coprocessor
  };

  Accelerator(Kind kind) : kind(kind) {}
  virtual ~Accelerator();

  /// Getter for the kind of this accelerator.
  Kind getKind() const { return kind; }

  static std::vector<Accelerator *> getAcceleratorList();
  virtual bool isActive() const = 0;
  virtual void prepareAccelerator(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::MLIRContext &context, mlir::PassManager &pm,
      onnx_mlir::EmissionTargetType &emissionTarget) const = 0;

protected:
  // static llvm::SmallPtrSet<Accelerator *, 2> acceleratorTargets;
  static std::vector<Accelerator *> acceleratorTargets;

  /// Kind of accelerator.
  Kind kind;
};

} // namespace accel
} // namespace onnx_mlir
