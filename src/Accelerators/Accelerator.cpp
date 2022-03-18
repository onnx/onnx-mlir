/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Accelerator.cpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Accelerator base class.
//
// To add support for a new accelerator kind include the accelerator header and
// provide a weak definition for the getInstance() member function.
//===----------------------------------------------------------------------===//

#include "src/Accelerators/Accelerator.hpp"
#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nnpa"

namespace onnx_mlir {
namespace accel {

llvm::SmallPtrSet<Accelerator *, 2> Accelerator::accelerators;

// Provide a weak definition for the getInstance() member function. A strong
// definition (to override this one) must be provided by the accelerator
// library.
__attribute__((weak, noinline)) Accelerator *
nnpa::NNPAAccelerator::getInstance() {
  LLVM_DEBUG(
      llvm::dbgs()
          << "Using weak definition for NNPAAccelerator::getInstance()\n";);
  return nullptr;
}

void Accelerator::create(Accelerator::Kind kind,
    mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::MLIRContext &context,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType emissionTarget) {
  Accelerator *accel = nullptr;
  switch (kind) {
  case Kind::NNPA:
    accel = nnpa::NNPAAccelerator::getInstance();
    if (!accel)
      llvm::errs() << "NNPA accelerator not supported\n";
    break;
  }
  assert(accel && "Accelerator not initialized correctly");

  // Initialize the new accelerator and add it to the list of available ones.
  accel->prepare(module, context, pm, emissionTarget);
  accelerators.insert(accel);
}

Accelerator::~Accelerator() { accelerators.erase(this); }

const llvm::SmallPtrSetImpl<Accelerator *> &Accelerator::getAccelerators() {
  return accelerators;
}

} // namespace accel
} // namespace onnx_mlir
