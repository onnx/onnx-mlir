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
#include "src/Accelerators/Accelerators.inc"
#include "llvm/ADT/SmallVector.h"

// Define the macros used to generate various accelerators artifacts (via the
// use of the APPLY_TO_ACCELERATORS macro, which is defined in the cmake
// generated file Accelerators.inc).
#define CREATE_ACCEL_ENUM(name) name,
#define DECLARE_ACCEL_INIT_FUNCTION(name) extern void create##name();
#define INVOKE_ACCEL_INIT_FUNCTION(name) create##name();
#define CREATE_ACCEL_CL_ENUM(name)                                             \
  clEnumValN(accel::Accelerator::Kind::name, #name, #name " accelerator"),

namespace onnx_mlir {
namespace accel {

extern void initAccelerators();

class Accelerator {
public:
  /// Kinds of accelerators.
  enum class Kind {
    // clang-format off
    APPLY_TO_ACCELERATORS(CREATE_ACCEL_ENUM) 
    NONE
    // clang-format on
  };

  Accelerator(Kind kind) : kind(kind) {}
  virtual ~Accelerator() = default;

  /// Getter for the kind of this accelerator.
  Kind getKind() const { return kind; }

  /// Returns the set of accelerators available.
  static const llvm::SmallVectorImpl<Accelerator *> &getAccelerators();

  /// Returns whether the accelerator is active.
  virtual bool isActive() const = 0;

  /// Load the MLIR dialects necessary to generate code for an accelerator.
  virtual void getOrLoadDialects(mlir::MLIRContext &context) const = 0;

  /// Add the transformations necessary to support the accelerator.
  virtual void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::PassManager &pm,
      onnx_mlir::EmissionTargetType &emissionTarget) const = 0;

  /// Register the MLIR dialects required to support an accelerator.
  virtual void registerDialects(mlir::DialectRegistry &registry) const = 0;

  /// Initialize the transformation passes required to generate code for an
  /// accelerator.
  virtual void initPasses(int optLevel) const = 0;

protected:
  static llvm::SmallVector<Accelerator *, 4> acceleratorTargets;
  /// Kind of accelerator.
  Kind kind;
};

} // namespace accel
} // namespace onnx_mlir
