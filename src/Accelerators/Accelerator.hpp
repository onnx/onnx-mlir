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
#include "llvm/ADT/SmallPtrSet.h"

namespace onnx_mlir {
namespace accel {

class Accelerator {
public:
  /// Kinds of accelerators.
  enum class Kind {
    NNPA, // IBM Telum coprocessor
  };

  virtual ~Accelerator();

  /// Getter for the kind of this accelerator.
  Kind getKind() const { return kind; }

  /// Create a new accelerator based on \p kind, initialize it, and add it to
  /// the list of available accelerators.
  static void create(Accelerator::Kind kind,
      mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::MLIRContext &context,
      mlir::PassManager &pm, onnx_mlir::EmissionTargetType emissionTarget);

  /// Return the list of available accelerators.
  static const llvm::SmallPtrSetImpl<Accelerator *> &getAccelerators();

protected:
  Accelerator(Kind kind) : kind(kind) {}

private:
  /// Accelerators available.
  static llvm::SmallPtrSet<Accelerator *, 2> accelerators;

  /// Initialize the accelerator.
  virtual void prepare(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::MLIRContext &context, mlir::PassManager &pm,
      onnx_mlir::EmissionTargetType emissionTarget) const = 0;

  /// Kind of accelerator.
  Kind kind;
};

} // namespace accel
} // namespace onnx_mlir
